"""
evaluator.py - Offline routing evaluation scenarios.

This gives BYOK a lightweight quality gate: representative agent tasks should
route to models with the expected capability, stay under budget, and respect
privacy/tool constraints without making paid API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

from byok.core.classifier import TaskClassifier
from byok.core.policy import RoutingPolicy
from byok.core.registry import ModelRegistry
from byok.core.router import ModelRouter, RoutingDecision
from byok.core.token_budget import TokenBudget, TokenBudgeter
from byok.storage.spend_tracker import SpendTracker


@dataclass
class EvalScenario:
    name: str
    message: str
    agent: Optional[str] = None
    task: Optional[str] = None
    mode: Optional[str] = None
    tools: bool = False
    privacy: bool = False
    max_cost_usd: Optional[float] = None
    max_output_tokens: Optional[int] = None
    expected_task: Optional[str] = None
    expected_strength: Optional[str] = None
    requires_tools: bool = False
    requires_local: bool = False


@dataclass
class EvalResult:
    scenario: EvalScenario
    passed: bool
    errors: list[str] = field(default_factory=list)
    selected_model: Optional[str] = None
    task_type: Optional[str] = None
    mode: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reason: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count


def load_scenarios(path: str | Path) -> list[EvalScenario]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    scenarios = []
    for entry in data.get("scenarios", []):
        scenarios.append(EvalScenario(**entry))
    return scenarios


class RoutingEvaluator:
    """Run offline routing scenarios against the current model/policy config."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: RoutingPolicy,
        spend_tracker: SpendTracker,
    ):
        self.registry = registry
        self.policy = policy
        self.spend_tracker = spend_tracker
        self.classifier = TaskClassifier()
        self.token_budgeter = TokenBudgeter()

    def run(self, scenarios: list[EvalScenario]) -> EvalSummary:
        return EvalSummary([self.evaluate(scenario) for scenario in scenarios])

    def evaluate(self, scenario: EvalScenario) -> EvalResult:
        messages = [{"role": "user", "content": scenario.message}]
        if scenario.privacy:
            messages.insert(0, {"role": "system", "content": "Keep this private and local."})

        tools = [{"type": "function", "function": {"name": "search"}}] if scenario.tools else []
        task = self.classifier.classify(messages, tools)

        if scenario.agent:
            task = replace(task, agent_role=scenario.agent)
        if scenario.task:
            task = replace(task, task_type=scenario.task)
        elif task.agent_role:
            policy_task = self.policy.task_for_agent(task.agent_role)
            if policy_task:
                task = replace(task, task_type=policy_task)
        if scenario.privacy:
            task = replace(task, privacy_required=True)

        controls = self.policy.controls_for(
            agent_role=task.agent_role,
            explicit_mode=scenario.mode,
            explicit_max_cost_usd=scenario.max_cost_usd,
            explicit_max_output_tokens=scenario.max_output_tokens,
        )
        router = ModelRouter(self.registry, self.spend_tracker, mode=controls.mode)
        decision = router.route(
            task,
            max_cost_usd=controls.max_cost_usd,
            requested_max_tokens=controls.max_output_tokens,
        )

        if decision is None:
            return EvalResult(
                scenario=scenario,
                passed=False,
                errors=["no model available"],
                task_type=task.task_type,
                mode=controls.mode,
            )

        token_budget = self.token_budgeter.budget_for_model_cost(
            task=task,
            cost_per_1k_input=decision.selected_model.cost_per_1k_input,
            cost_per_1k_output=decision.selected_model.cost_per_1k_output,
            mode=controls.mode,
            requested_max_tokens=controls.max_output_tokens,
            max_cost_usd=controls.max_cost_usd,
        )
        errors = self._validate(scenario, task.task_type, decision, token_budget, controls.max_cost_usd)

        return EvalResult(
            scenario=scenario,
            passed=not errors,
            errors=errors,
            selected_model=decision.selected_model.name,
            task_type=task.task_type,
            mode=controls.mode,
            estimated_cost_usd=decision.estimated_cost_usd,
            max_output_tokens=token_budget.max_output_tokens,
            reason=decision.reason,
        )

    def _validate(
        self,
        scenario: EvalScenario,
        task_type: str,
        decision: RoutingDecision,
        token_budget: TokenBudget,
        max_cost_usd: Optional[float],
    ) -> list[str]:
        errors: list[str] = []
        model = decision.selected_model

        if scenario.expected_task and task_type != scenario.expected_task:
            errors.append(f"expected task {scenario.expected_task}, got {task_type}")

        if scenario.expected_strength and scenario.expected_strength not in model.strengths:
            errors.append(f"selected model lacks strength {scenario.expected_strength}")

        if scenario.requires_tools and not model.supports_tools:
            errors.append("selected model does not support tools")

        if scenario.requires_local and not model.local:
            errors.append("selected model is not local")

        if max_cost_usd is not None and decision.estimated_cost_usd > max_cost_usd + 1e-9:
            errors.append(f"estimated cost ${decision.estimated_cost_usd:.6f} exceeds ${max_cost_usd:.6f}")

        if token_budget.max_output_tokens <= 0:
            errors.append("token budget leaves no output tokens")

        return errors


class ConfiguredModelRegistry:
    """
    Registry adapter for offline evaluation.

    It includes enabled models from models.yaml even if their API keys are not
    present. This lets contributors evaluate routing plans without spending
    money or configuring paid providers.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def available_models(self):
        return [model for model in self.registry.all_models() if model.enabled]
