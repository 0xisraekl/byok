"""
router.py — Smart Model Router

BYOK's job is not to pick the cheapest model or the fanciest model.
It picks the cheapest model that is likely to do the task well, and it
explains the trade-off in plain English.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from byok.core.classifier import TaskProfile
from byok.core.registry import ModelConfig, ModelRegistry
from byok.core.token_budget import TokenBudgeter
from byok.storage.spend_tracker import SpendTracker


@dataclass
class RoutingDecision:
    """
    The output of the router: which model to use and why.
    Logged on every request so users can see what BYOK decided.
    """

    selected_model: ModelConfig
    score: float
    reason: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_output_tokens: int = 500
    quality_estimate: float = 0.0
    best_quality_model: Optional[str] = None
    premium_reference_cost_usd: Optional[float] = None
    estimated_savings_usd: Optional[float] = None
    estimated_savings_pct: Optional[float] = None

    def __str__(self) -> str:
        savings = ""
        if self.estimated_savings_usd is not None and self.estimated_savings_usd > 0:
            savings = f", saved ~${self.estimated_savings_usd:.5f}"
        return (
            f"→ {self.selected_model.name}  "
            f"(score: {self.score:.1f}, quality: {self.quality_estimate:.0%}, "
            f"est. cost: ${self.estimated_cost_usd:.5f}{savings})\n"
            f"  reason: {self.reason}"
        )


class ModelRouter:
    """
    Smart router for choosing the best model in the user's BYOK pool.

    Modes:
      balanced  = best default; quality first, cost-aware
      cheap     = cheapest good-enough model
      quality   = strongest specialist model under budget
      speed     = low-latency specialist
      private   = local-only

    Hard filters:
      ✗ model disabled/no key (handled by registry)
      ✗ model is at monthly spend limit
      ✗ task needs tools but model does not support tools
      ✗ context is too large
      ✗ privacy required but model is not local
    """

    GOOD_ENOUGH_BY_DIFFICULTY = {
        "easy": 0.58,
        "medium": 0.70,
        "hard": 0.82,
    }

    OUTPUT_TOKEN_ESTIMATES = {
        "simple_chat": {"easy": 120, "medium": 250, "hard": 500},
        "extraction": {"easy": 250, "medium": 650, "hard": 1200},
        "summarization": {"easy": 250, "medium": 700, "hard": 1400},
        "writing": {"easy": 350, "medium": 900, "hard": 1800},
        "coding": {"easy": 500, "medium": 1400, "hard": 2800},
        "reasoning": {"easy": 400, "medium": 1100, "hard": 2200},
        "math": {"easy": 300, "medium": 900, "hard": 1800},
        "data_analysis": {"easy": 500, "medium": 1400, "hard": 2600},
        "tool_calling": {"easy": 250, "medium": 700, "hard": 1400},
    }

    def __init__(self, registry: ModelRegistry, spend_tracker: SpendTracker, mode: str = "balanced"):
        self.registry = registry
        self.spend_tracker = spend_tracker
        self.mode = mode
        self.token_budgeter = TokenBudgeter()

    def route(
        self,
        task: TaskProfile,
        max_cost_usd: Optional[float] = None,
        requested_max_tokens: Optional[int] = None,
    ) -> Optional[RoutingDecision]:
        candidates = self.registry.available_models()
        if not candidates:
            return None

        scored: list[tuple[ModelConfig, float, str, float, float]] = []
        for model in candidates:
            score, reason = self._score(model, task, max_cost_usd, requested_max_tokens)
            if score == float("-inf"):
                continue
            quality = self._quality_for(model, task)
            cost = self._estimate_cost(model, task, max_cost_usd, requested_max_tokens)
            scored.append((model, score, reason, quality, cost))

        if not scored:
            return None

        scored.sort(key=lambda x: (-x[1], x[4], x[0].priority))
        quality_sorted = sorted(scored, key=lambda x: (-x[3], x[4], x[0].priority))

        # Balanced mode is the product wedge: do not blindly pick the fanciest
        # model if a much cheaper model is nearly as strong for this exact task.
        # This keeps "best results" and "lower cost" working together.
        if self.mode == "balanced" and len(scored) > 1:
            top_quality = quality_sorted[0][3]
            top_cost = quality_sorted[0][4]
            quality_floor = max(self.GOOD_ENOUGH_BY_DIFFICULTY.get(task.difficulty, 0.70), top_quality - 0.05)
            near_best_value = [x for x in scored if x[3] >= quality_floor and (top_cost <= 0 or x[4] <= top_cost * 0.75)]
            if near_best_value:
                near_best_value.sort(key=lambda x: (x[4], -x[3], x[0].priority))
                best_model, best_score, best_reason, best_quality, best_cost = near_best_value[0]
                best_reason += ", balanced mode: near-best quality at lower cost"
            else:
                best_model, best_score, best_reason, best_quality, best_cost = scored[0]
        else:
            best_model, best_score, best_reason, best_quality, best_cost = scored[0]

        best_quality_model, _, _, top_quality, premium_cost = quality_sorted[0]
        savings_usd = max(premium_cost - best_cost, 0.0)
        savings_pct = (savings_usd / premium_cost * 100) if premium_cost > 0 else 0.0

        alternatives = [(m.name, s) for m, s, _, _, _ in scored if m.name != best_model.name][:3]
        selected_budget = self.token_budgeter.budget_for_model_cost(
            task=task,
            cost_per_1k_input=best_model.cost_per_1k_input,
            cost_per_1k_output=best_model.cost_per_1k_output,
            mode=self.mode,
            requested_max_tokens=requested_max_tokens,
            max_cost_usd=max_cost_usd,
        )

        return RoutingDecision(
            selected_model=best_model,
            score=best_score,
            reason=best_reason,
            alternatives=alternatives,
            estimated_cost_usd=best_cost,
            estimated_output_tokens=selected_budget.max_output_tokens,
            quality_estimate=best_quality,
            best_quality_model=best_quality_model.name,
            premium_reference_cost_usd=premium_cost,
            estimated_savings_usd=savings_usd,
            estimated_savings_pct=savings_pct,
        )

    def _score(
        self,
        model: ModelConfig,
        task: TaskProfile,
        max_cost_usd: Optional[float] = None,
        requested_max_tokens: Optional[int] = None,
    ) -> tuple[float, str]:
        reasons: list[str] = []
        token_budget = self.token_budgeter.budget_for_model_cost(
            task=task,
            cost_per_1k_input=model.cost_per_1k_input,
            cost_per_1k_output=model.cost_per_1k_output,
            mode=self.mode,
            requested_max_tokens=requested_max_tokens,
            max_cost_usd=max_cost_usd,
        )
        cost = self._estimate_cost(model, task, max_cost_usd, requested_max_tokens)

        # Hard filters
        if max_cost_usd is not None and token_budget.max_output_tokens <= 0:
            return float("-inf"), "request cost limit leaves no output token budget"

        if max_cost_usd is not None and cost > max_cost_usd:
            return float("-inf"), f"estimated cost ${cost:.5f} exceeds request limit ${max_cost_usd:.5f}"

        if model.spend_limit_monthly_usd > 0:
            monthly_spend = self.spend_tracker.get_monthly_spend(model.name)
            if monthly_spend >= model.spend_limit_monthly_usd:
                return float("-inf"), "at monthly spend limit"
            if monthly_spend + cost > model.spend_limit_monthly_usd:
                return (
                    float("-inf"),
                    f"estimated call would exceed monthly spend limit "
                    f"(${monthly_spend:.4f} + ${cost:.4f} > ${model.spend_limit_monthly_usd:.2f})",
                )

        if task.has_tools and not model.supports_tools:
            return float("-inf"), "doesn't support tool calling"

        if not model.can_handle_tokens(task.context_tokens):
            return float("-inf"), f"context too large ({task.context_tokens} > {model.context_window})"

        if (task.privacy_required or self.mode == "private") and not model.local:
            return float("-inf"), "privacy required, model is not local"

        quality = self._quality_for(model, task)
        score = 0.0

        # Capability / task-fit score
        if task.task_type in model.strengths:
            score += 18.0
            reasons.append(f"specialist for {task.task_type}")
        elif "general" in model.strengths:
            score += 7.0
            reasons.append("general-purpose fallback")
        else:
            score -= 3.0
            reasons.append("not a direct specialist")

        for secondary in task.secondary_types:
            if secondary in model.strengths:
                score += 5.0
                reasons.append(f"also handles {secondary}")

        if task.context_tokens > 30000 and "long_context" in model.strengths:
            score += 8.0
            reasons.append("long-context fit")
        elif task.context_tokens > 30000 and "long_context" not in model.strengths:
            score -= 6.0
            reasons.append("large input without long-context strength")

        # Quality prior matters because the user's goal is best result per prompt.
        quality_weight = {"cheap": 14.0, "balanced": 26.0, "quality": 38.0, "speed": 20.0, "private": 20.0}.get(self.mode, 26.0)
        score += quality * quality_weight
        reasons.append(f"quality prior {quality:.0%}")

        # Good-enough ladder: cheap mode can use cheaper models, but not terrible ones.
        good_enough = self.GOOD_ENOUGH_BY_DIFFICULTY.get(task.difficulty, 0.70)
        if self.mode == "cheap":
            good_enough -= 0.08
        elif self.mode == "quality":
            good_enough += 0.05

        if quality < good_enough:
            penalty = (good_enough - quality) * 35
            score -= penalty
            reasons.append(f"below {task.difficulty} quality floor")
        else:
            score += 4.0
            reasons.append("passes quality floor")

        # Cost pressure: logarithmic-ish bucket so cost matters without dominating quality.
        cost_penalty = self._cost_penalty(cost)
        cost_weight = {"cheap": 3.2, "balanced": 1.6, "quality": 0.6, "speed": 1.0, "private": 1.0}.get(self.mode, 1.6)
        score -= cost_penalty * cost_weight
        if cost == 0:
            reasons.append("free/local")
        elif self.mode == "cheap":
            reasons.append("cheap mode: cost optimized")

        # Latency preference
        if model.latency == "low":
            score += 3.0 if self.mode == "speed" else 1.0
            reasons.append("speed mode: fast" if self.mode == "speed" else "fast")
        elif model.latency == "high":
            score -= 4.0 if self.mode == "speed" else 1.0
            reasons.append("slower")

        if model.local:
            score += 4.0 if self.mode in ("cheap", "private") else 1.5
            reasons.append("local/private")

        # Difficulty guardrails: do not route hard work to tiny/cheap models unless configured as high-quality.
        if task.difficulty == "hard" and quality < 0.78:
            score -= 7.0
            reasons.append("hard-task guardrail")

        return score, ", ".join(reasons)

    def _quality_for(self, model: ModelConfig, task: TaskProfile) -> float:
        """Return a 0-1 quality prior for this model on this task."""
        if task.task_type in model.task_quality:
            base = float(model.task_quality[task.task_type])
        else:
            base = float(model.quality_score)
            if task.task_type in model.strengths:
                base += 0.06
            elif "general" not in model.strengths:
                base -= 0.06

        for secondary in task.secondary_types:
            if secondary in model.task_quality:
                base = max(base, float(model.task_quality[secondary]) - 0.03)
            elif secondary in model.strengths:
                base += 0.02

        if task.difficulty == "hard":
            base -= 0.02
        elif task.difficulty == "easy":
            base += 0.02

        return max(0.0, min(0.99, base))

    def _estimate_output_tokens(self, task: TaskProfile) -> int:
        return self.token_budgeter.estimate_output_tokens(task)

    def _estimate_cost(
        self,
        model: ModelConfig,
        task: TaskProfile,
        max_cost_usd: Optional[float] = None,
        requested_max_tokens: Optional[int] = None,
    ) -> float:
        # Use the mode-aware token cap for selection. This means cheap/balanced
        # modes do not merely pick cheaper models; they also account for the
        # lower completion budget BYOK will send to the provider.
        output_tokens = self.token_budgeter.budget_for_model_cost(
            task=task,
            cost_per_1k_input=model.cost_per_1k_input,
            cost_per_1k_output=model.cost_per_1k_output,
            mode=self.mode,
            requested_max_tokens=requested_max_tokens,
            max_cost_usd=max_cost_usd,
        ).max_output_tokens
        return (
            task.context_tokens * model.cost_per_1k_input / 1000
            + output_tokens * model.cost_per_1k_output / 1000
        )

    def _cost_penalty(self, estimated_cost: float) -> float:
        """Bucket cost so a $0.0002 vs $0.002 call matters less than $0.002 vs $0.20."""
        if estimated_cost <= 0:
            return 0.0
        if estimated_cost < 0.001:
            return 0.5
        if estimated_cost < 0.01:
            return 1.5
        if estimated_cost < 0.05:
            return 3.0
        if estimated_cost < 0.20:
            return 5.0
        return 8.0
