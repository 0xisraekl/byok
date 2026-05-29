"""Tests for proxy fallback helpers."""

from byok.core.registry import ModelConfig
from byok.core.router import RoutingDecision
from byok.core.classifier import TaskProfile
from byok.core.policy import RoutingPolicy
from byok.proxy.server import (
    _apply_byok_metadata,
    _apply_policy_task,
    _attempt_models,
    _combine_cost_limits,
    _estimate_cost_for_model,
    _has_explicit_task_hint,
    _mode_from_request,
    _mode_from_request_optional,
    _optional_float,
    _remaining_run_budget,
    _strip_byok_hints_from_messages,
    _token_budget_for_model,
)


def make_model(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="openai",
        model_id=name,
        strengths=["general"],
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        api_key_env=None,
    )


class FakeRegistry:
    def __init__(self, models):
        self.models = {m.name: m for m in models}

    def get(self, name: str):
        return self.models.get(name)


def test_attempt_models_returns_selected_then_ranked_alternatives():
    selected = make_model("selected")
    backup = make_model("backup")
    third = make_model("third")
    decision = RoutingDecision(
        selected_model=selected,
        score=10.0,
        reason="test",
        alternatives=[("backup", 9.0), ("third", 8.0)],
    )

    attempts = _attempt_models(decision, FakeRegistry([selected, backup, third]))

    assert [m.name for m in attempts] == ["selected", "backup", "third"]


def test_attempt_models_skips_duplicate_and_missing_alternatives():
    selected = make_model("selected")
    backup = make_model("backup")
    decision = RoutingDecision(
        selected_model=selected,
        score=10.0,
        reason="test",
        alternatives=[("selected", 9.5), ("missing", 9.0), ("backup", 8.5)],
    )

    attempts = _attempt_models(decision, FakeRegistry([selected, backup]))

    assert [m.name for m in attempts] == ["selected", "backup"]


def test_mode_from_request_reads_explicit_byok_mode():
    assert _mode_from_request("auto", "quality") == "quality"


def test_mode_from_request_optional_returns_none_without_request_mode():
    assert _mode_from_request_optional("auto", None) is None
    assert _mode_from_request_optional("auto:cheap", None) == "cheap"


def test_apply_byok_metadata_overrides_task_agent_and_privacy():
    task = TaskProfile(
        task_type="simple_chat",
        secondary_types=[],
        difficulty="easy",
        context_tokens=20,
        has_tools=False,
        privacy_required=False,
        confidence=0.8,
    )

    updated = _apply_byok_metadata(
        task,
        {"task": "coding", "difficulty": "hard", "agent": "coder", "privacy": True},
    )

    assert updated.task_type == "coding"
    assert updated.difficulty == "hard"
    assert updated.agent_role == "coder"
    assert updated.privacy_required is True
    assert updated.route_hints["task"] == "coding"


def test_policy_task_applies_when_request_has_no_task_hint():
    task = TaskProfile(
        task_type="simple_chat",
        secondary_types=[],
        difficulty="easy",
        context_tokens=20,
        has_tools=False,
        privacy_required=False,
        confidence=0.8,
        agent_role="coder",
    )
    policy = RoutingPolicy("config/routing_policy.yaml")

    updated = _apply_policy_task(task, policy, explicit_task=False)

    assert updated.task_type == "coding"


def test_policy_task_does_not_override_explicit_task_hint():
    task = TaskProfile(
        task_type="math",
        secondary_types=[],
        difficulty="easy",
        context_tokens=20,
        has_tools=False,
        privacy_required=False,
        confidence=0.8,
        agent_role="coder",
        route_hints={"task": "math"},
    )
    policy = RoutingPolicy("config/routing_policy.yaml")

    updated = _apply_policy_task(task, policy, explicit_task=_has_explicit_task_hint({}, task))

    assert updated.task_type == "math"


def test_strip_byok_hints_before_provider_forwarding():
    messages = [
        {"role": "user", "content": "[byok:task=coding,agent=coder] Handle this."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please solve. [byok:privacy=true]"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
    ]

    cleaned = _strip_byok_hints_from_messages(messages)

    assert cleaned[0]["content"] == "Handle this."
    assert cleaned[1]["content"][0]["text"] == "Please solve."
    assert cleaned[1]["content"][1] == messages[1]["content"][1]


def test_optional_float_parses_request_budget_values():
    assert _optional_float("0.0025") == 0.0025
    assert _optional_float(1) == 1.0
    assert _optional_float("-1") is None
    assert _optional_float("not-a-number") is None


def test_run_budget_helpers_compute_effective_limit():
    assert _remaining_run_budget(0.006, 0.010) == 0.004
    assert _remaining_run_budget(0.012, 0.010) == 0.0
    assert _remaining_run_budget(0.006, None) is None

    assert _combine_cost_limits(None, None) is None
    assert _combine_cost_limits(0.004, None) == 0.004
    assert _combine_cost_limits(0.004, 0.002) == 0.002


def test_fallback_token_budget_is_recomputed_for_each_model_cost():
    task = TaskProfile(
        task_type="coding",
        secondary_types=[],
        difficulty="medium",
        context_tokens=1000,
        has_tools=False,
        privacy_required=False,
        confidence=0.9,
    )
    cheap_primary = make_model("cheap-primary")
    expensive_fallback = make_model("expensive-fallback")
    expensive_fallback.cost_per_1k_output = 0.01

    primary_budget = _token_budget_for_model(
        cheap_primary,
        task,
        mode="quality",
        requested_max_tokens=1400,
        max_cost_usd=0.006,
    )
    fallback_budget = _token_budget_for_model(
        expensive_fallback,
        task,
        mode="quality",
        requested_max_tokens=1400,
        max_cost_usd=0.006,
    )

    assert primary_budget.max_output_tokens == 1400
    assert fallback_budget.max_output_tokens == 500
    assert _estimate_cost_for_model(expensive_fallback, task, fallback_budget.max_output_tokens) <= 0.006
