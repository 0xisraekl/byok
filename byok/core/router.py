"""
router.py — Model Router

Takes a TaskProfile (what kind of task) + the model registry (what's available)
and decides which model in YOUR pool to use.

Decision process:
  1. Filter to models that CAN handle the task (tools, context, privacy, spend)
  2. Score each candidate by how well it fits
  3. Pick the highest scorer
  4. If nothing is available, try fallback chain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from byok.core.classifier import TaskProfile
from byok.core.registry import ModelConfig, ModelRegistry
from byok.storage.spend_tracker import SpendTracker


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    The output of the router: which model to use and why.
    This is logged on every request so you can see what BYOK decided.
    """
    selected_model: ModelConfig
    score: float
    reason: str                                  # human-readable explanation
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    estimated_cost_usd: float = 0.0

    def __str__(self) -> str:
        return (
            f"→ {self.selected_model.name}  "
            f"(score: {self.score:.1f}, "
            f"est. cost: ${self.estimated_cost_usd:.5f})\n"
            f"  reason: {self.reason}"
        )


# ─────────────────────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Scores and selects the best model from your pool for each task.

    Scoring breakdown:
      +10  per strength that exactly matches the task type
      + 5  per secondary type that matches a strength
      + 3  local model bonus (free = great)
      + 2  low latency bonus (if scoring favors speed)
      - 2  high latency penalty
      + 0  model priority (used only as tiebreaker)

    Hard filters (score → -∞ if failed):
      ✗  model is at its monthly spend limit
      ✗  task needs tools but model doesn't support them
      ✗  message is too long for the model's context window
      ✗  privacy required but model is not local
    """

    def __init__(self, registry: ModelRegistry, spend_tracker: SpendTracker):
        self.registry = registry
        self.spend_tracker = spend_tracker

    def route(self, task: TaskProfile) -> Optional[RoutingDecision]:
        """
        Main entry point. Returns the best RoutingDecision, or None if
        no model in the pool can handle this task.
        """
        candidates = self.registry.available_models()

        if not candidates:
            return None

        # Score every candidate
        scored: list[tuple[ModelConfig, float, str]] = []
        for model in candidates:
            score, reason = self._score(model, task)
            if score > float("-inf"):
                scored.append((model, score, reason))

        if not scored:
            return None

        # Sort by score descending, then priority ascending (lower = better)
        scored.sort(key=lambda x: (-x[1], x[0].priority))

        best_model, best_score, best_reason = scored[0]

        # Estimate cost for logging
        avg_output_tokens = 500  # rough guess
        estimated_cost = (
            task.context_tokens * best_model.cost_per_1k_input / 1000
            + avg_output_tokens * best_model.cost_per_1k_output / 1000
        )

        # Build the alternatives list (for logging/transparency)
        alternatives = [
            (m.name, s) for m, s, _ in scored[1:4]  # top 3 runners-up
        ]

        return RoutingDecision(
            selected_model=best_model,
            score=best_score,
            reason=best_reason,
            alternatives=alternatives,
            estimated_cost_usd=estimated_cost,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, model: ModelConfig, task: TaskProfile) -> tuple[float, str]:
        """
        Score a single model against the task.
        Returns (score, reason_string).
        A score of -inf means this model is disqualified.
        """
        reasons: list[str] = []
        score = 0.0

        # ── Hard filters (instant disqualify) ─────────────────────────────

        # Check spend limit
        if model.spend_limit_monthly_usd > 0:
            monthly_spend = self.spend_tracker.get_monthly_spend(model.name)
            if monthly_spend >= model.spend_limit_monthly_usd:
                return float("-inf"), "at monthly spend limit"

        # Check tool support
        if task.has_tools and not model.supports_tools:
            return float("-inf"), "doesn't support tool calling"

        # Check context window
        if not model.can_handle_tokens(task.context_tokens):
            return float("-inf"), f"context too large ({task.context_tokens} > {model.context_window})"

        # Privacy: must use local model
        if task.privacy_required and not model.local:
            return float("-inf"), "privacy required, model is not local"

        # ── Positive scoring ───────────────────────────────────────────────

        # Primary strength match — the most important factor
        if task.task_type in model.strengths:
            score += 10.0
            reasons.append(f"strong at {task.task_type}")

        # Secondary strength matches
        for secondary in task.secondary_types:
            if secondary in model.strengths:
                score += 5.0
                reasons.append(f"also good at {secondary}")

        # Local model bonus (it's free!)
        if model.local:
            score += 3.0
            reasons.append("local/free")

        # Latency scoring
        if model.latency == "low":
            score += 2.0
            reasons.append("fast")
        elif model.latency == "high":
            score -= 2.0

        # Difficulty vs. model capability:
        # For "hard" tasks, prefer models that aren't the cheapest/simplest
        if task.difficulty == "hard" and model.cost_per_1k_input < 0.0005 and not model.local:
            score -= 3.0  # cheap cloud models probably can't handle hard tasks

        # If the task is simple, don't waste an expensive model
        if task.difficulty == "easy" and model.cost_per_1k_input > 0.002:
            score -= 2.0
            reasons.append("overkill for easy task")

        # If no specific strength matched at all, small penalty
        if task.task_type not in model.strengths:
            score -= 1.0

        reason = ", ".join(reasons) if reasons else "general capability"
        return score, reason
