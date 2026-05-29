"""Tests for offline routing evaluation scenarios."""

from byok.core.evaluator import ConfiguredModelRegistry, EvalScenario, RoutingEvaluator, load_scenarios
from byok.core.policy import RoutingPolicy
from byok.core.registry import ModelConfig
from byok.storage.spend_tracker import SpendTracker


class FakeRegistry:
    def __init__(self, models):
        self._models = models

    def available_models(self):
        return [m for m in self._models if m.enabled]


def make_model(name: str, strengths: list[str], **overrides) -> ModelConfig:
    defaults = dict(
        name=name,
        provider="test",
        model_id=name,
        strengths=strengths,
        context_window=128000,
        supports_tools=False,
        local=False,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
        quality_score=0.80,
        task_quality={strengths[0]: 0.85} if strengths else {},
        enabled=True,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_load_scenarios_from_default_config():
    scenarios = load_scenarios("config/eval_scenarios.yaml")

    assert len(scenarios) >= 5
    assert any(s.name == "coder_subagent_small_budget" for s in scenarios)


def test_evaluator_passes_matching_coding_scenario():
    coder = make_model("coder", ["coding"], task_quality={"coding": 0.90})
    evaluator = RoutingEvaluator(
        registry=FakeRegistry([coder]),
        policy=RoutingPolicy("config/routing_policy.yaml"),
        spend_tracker=SpendTracker(":memory:"),
    )

    summary = evaluator.run([
        EvalScenario(
            name="coding",
            message="Implement this function",
            agent="coder",
            expected_task="coding",
            expected_strength="coding",
            max_cost_usd=0.01,
        )
    ])

    assert summary.passed is True
    assert summary.passed_count == 1
    assert summary.results[0].selected_model == "coder"


def test_evaluator_fails_when_selected_model_lacks_expected_strength():
    weak = make_model("weak-chat", ["simple_chat"])
    evaluator = RoutingEvaluator(
        registry=FakeRegistry([weak]),
        policy=RoutingPolicy("config/routing_policy.yaml"),
        spend_tracker=SpendTracker(":memory:"),
    )

    result = evaluator.evaluate(
        EvalScenario(
            name="bad-coding",
            message="Implement this function",
            task="coding",
            expected_task="coding",
            expected_strength="coding",
            max_cost_usd=0.01,
        )
    )

    assert result.passed is False
    assert any("lacks strength coding" in error for error in result.errors)


def test_configured_registry_includes_enabled_models_without_keys():
    keyed = make_model("cloud-without-key", ["coding"], api_key_env="MISSING_TEST_KEY")
    disabled = make_model("disabled", ["coding"], enabled=False)

    class Registry:
        def all_models(self):
            return [keyed, disabled]

    configured = ConfiguredModelRegistry(Registry())

    assert [m.name for m in configured.available_models()] == ["cloud-without-key"]
