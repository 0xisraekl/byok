"""
Tests for the model router.
Run with:  .venv/bin/pytest tests/ -v
"""

import pytest
from byok.core.classifier import TaskProfile
from byok.core.registry import ModelConfig, ModelRegistry
from byok.core.router import ModelRouter
from byok.storage.spend_tracker import SpendTracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def spend_tracker():
    """In-memory spend tracker — no file created during tests."""
    return SpendTracker(":memory:")


def make_model(**overrides) -> ModelConfig:
    """Build a ModelConfig with sensible defaults, overridden by kwargs."""
    defaults = dict(
        name="test-model",
        provider="openai",
        model_id="gpt-test",
        strengths=["coding"],
        context_window=128000,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        latency="medium",
        supports_tools=True,
        local=False,
        spend_limit_monthly_usd=0,
        enabled=True,
        priority=1,
        api_key_env=None,
    )
    defaults.update(overrides)
    m = ModelConfig(**defaults)
    # Bypass key check for tests (no real .env needed)
    m.__class__.has_valid_key = property(lambda self: True)
    return m


def make_profile(**overrides) -> TaskProfile:
    """Build a TaskProfile with sensible defaults."""
    defaults = dict(
        task_type="coding",
        secondary_types=[],
        difficulty="medium",
        context_tokens=500,
        has_tools=False,
        privacy_required=False,
        confidence=0.9,
    )
    defaults.update(overrides)
    return TaskProfile(**defaults)


class FakeRegistry:
    """Minimal registry that returns a fixed list of models."""
    def __init__(self, models: list[ModelConfig]):
        self._models = models

    def available_models(self):
        return [m for m in self._models if m.enabled]

    def all_models(self):
        return list(self._models)


# ── Basic routing ─────────────────────────────────────────────────────────────

class TestBasicRouting:

    def test_routes_to_matching_strength(self, spend_tracker):
        coding_model = make_model(name="coder", strengths=["coding"])
        chat_model   = make_model(name="chatter", strengths=["simple_chat"])
        registry = FakeRegistry([coding_model, chat_model])
        router = ModelRouter(registry, spend_tracker)

        decision = router.route(make_profile(task_type="coding"))

        assert decision is not None
        assert decision.selected_model.name == "coder"

    def test_returns_none_when_no_models(self, spend_tracker):
        registry = FakeRegistry([])
        router = ModelRouter(registry, spend_tracker)
        decision = router.route(make_profile())
        assert decision is None

    def test_decision_has_reason(self, spend_tracker):
        model = make_model(name="m1", strengths=["coding"])
        router = ModelRouter(FakeRegistry([model]), spend_tracker)
        decision = router.route(make_profile(task_type="coding"))
        assert decision is not None
        assert len(decision.reason) > 0

    def test_decision_has_estimated_cost(self, spend_tracker):
        model = make_model(name="m1", cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        router = ModelRouter(FakeRegistry([model]), spend_tracker)
        decision = router.route(make_profile(context_tokens=1000))
        assert decision is not None
        assert decision.estimated_cost_usd >= 0


# ── Spend limit enforcement ────────────────────────────────────────────────────

class TestSpendLimits:

    def test_model_at_spend_limit_is_skipped(self, spend_tracker):
        """If model A is at its limit and model B is not, B should be chosen."""
        expensive = make_model(name="expensive", strengths=["coding"], spend_limit_monthly_usd=1.0)
        cheap     = make_model(name="cheap", strengths=["coding"], spend_limit_monthly_usd=0.0)
        registry = FakeRegistry([expensive, cheap])
        router = ModelRouter(registry, spend_tracker)

        # Simulate expensive being at its limit
        spend_tracker.log(
            model_name="expensive", provider="openai", task_type="coding",
            difficulty="medium", input_tokens=1000, output_tokens=500,
            cost_usd=1.0,  # exactly at the $1 limit
            routing_reason="test",
        )

        decision = router.route(make_profile(task_type="coding"))
        assert decision is not None
        assert decision.selected_model.name == "cheap"

    def test_model_under_limit_is_available(self, spend_tracker):
        model = make_model(name="m1", spend_limit_monthly_usd=10.0)
        spend_tracker.log(
            model_name="m1", provider="openai", task_type="coding",
            difficulty="easy", input_tokens=100, output_tokens=50,
            cost_usd=0.50,  # half the limit
            routing_reason="test",
        )
        router = ModelRouter(FakeRegistry([model]), spend_tracker)
        decision = router.route(make_profile())
        assert decision is not None
        assert decision.selected_model.name == "m1"

    def test_no_limit_model_always_available(self, spend_tracker):
        model = make_model(name="unlimited", spend_limit_monthly_usd=0.0)
        # Simulate large spend — shouldn't matter since limit is 0
        spend_tracker.log(
            model_name="unlimited", provider="openai", task_type="coding",
            difficulty="hard", input_tokens=100000, output_tokens=50000,
            cost_usd=999.99,
            routing_reason="test",
        )
        router = ModelRouter(FakeRegistry([model]), spend_tracker)
        decision = router.route(make_profile())
        assert decision is not None


# ── Privacy enforcement ────────────────────────────────────────────────────────

class TestPrivacyRouting:

    def test_privacy_forces_local_model(self, spend_tracker):
        cloud_model = make_model(name="cloud", local=False, strengths=["reasoning"])
        local_model = make_model(name="local", local=True,  strengths=["reasoning"])
        registry = FakeRegistry([cloud_model, local_model])
        router = ModelRouter(registry, spend_tracker)

        decision = router.route(make_profile(privacy_required=True))

        assert decision is not None
        assert decision.selected_model.name == "local"

    def test_privacy_returns_none_if_no_local_model(self, spend_tracker):
        cloud_only = make_model(name="cloud", local=False)
        registry = FakeRegistry([cloud_only])
        router = ModelRouter(registry, spend_tracker)

        decision = router.route(make_profile(privacy_required=True))
        assert decision is None


# ── Tool calling enforcement ───────────────────────────────────────────────────

class TestToolCallingRouting:

    def test_model_without_tools_skipped_when_tools_needed(self, spend_tracker):
        no_tools  = make_model(name="no-tools",   supports_tools=False, strengths=["tool_calling"])
        has_tools = make_model(name="has-tools",  supports_tools=True,  strengths=["tool_calling"])
        registry = FakeRegistry([no_tools, has_tools])
        router = ModelRouter(registry, spend_tracker)

        decision = router.route(make_profile(has_tools=True, task_type="tool_calling"))

        assert decision is not None
        assert decision.selected_model.name == "has-tools"

    def test_no_tool_needed_both_eligible(self, spend_tracker):
        no_tools  = make_model(name="no-tools",  supports_tools=False, strengths=["coding"])
        has_tools = make_model(name="has-tools", supports_tools=True,  strengths=["coding"])
        registry = FakeRegistry([no_tools, has_tools])
        router = ModelRouter(registry, spend_tracker)

        decision = router.route(make_profile(has_tools=False, task_type="coding"))
        assert decision is not None  # either is fine


# ── Context window enforcement ────────────────────────────────────────────────

class TestContextWindowRouting:

    def test_model_with_small_context_skipped_for_large_input(self, spend_tracker):
        small_ctx = make_model(name="small", context_window=4096,   strengths=["summarization"])
        large_ctx = make_model(name="large", context_window=200000, strengths=["summarization"])
        registry = FakeRegistry([small_ctx, large_ctx])
        router = ModelRouter(registry, spend_tracker)

        # Request needs more context than small model supports
        decision = router.route(make_profile(context_tokens=10000, task_type="summarization"))

        assert decision is not None
        assert decision.selected_model.name == "large"

    def test_small_context_request_uses_any_model(self, spend_tracker):
        small_ctx = make_model(name="small", context_window=4096)
        router = ModelRouter(FakeRegistry([small_ctx]), spend_tracker)

        decision = router.route(make_profile(context_tokens=500))
        assert decision is not None


# ── Scoring logic ──────────────────────────────────────────────────────────────

class TestRoutingModes:

    def test_private_mode_forces_local_model_even_without_privacy_keywords(self, spend_tracker):
        cloud_model = make_model(name="cloud", local=False, strengths=["simple_chat"])
        local_model = make_model(name="local", local=True, strengths=["simple_chat"])
        router = ModelRouter(FakeRegistry([cloud_model, local_model]), spend_tracker, mode="private")

        decision = router.route(make_profile(task_type="simple_chat", privacy_required=False))

        assert decision is not None
        assert decision.selected_model.name == "local"

    def test_cheap_mode_prefers_cheaper_model_when_models_are_otherwise_similar(self, spend_tracker):
        expensive = make_model(name="expensive", strengths=["summarization"], cost_per_1k_input=0.003)
        cheap = make_model(name="cheap", strengths=["summarization"], cost_per_1k_input=0.0001)
        router = ModelRouter(FakeRegistry([expensive, cheap]), spend_tracker, mode="cheap")

        decision = router.route(make_profile(task_type="summarization", difficulty="medium"))

        assert decision is not None
        assert decision.selected_model.name == "cheap"
        assert "cheap mode" in decision.reason

    def test_speed_mode_prefers_low_latency_model_when_models_are_otherwise_similar(self, spend_tracker):
        slow = make_model(name="slow", strengths=["simple_chat"], latency="high")
        fast = make_model(name="fast", strengths=["simple_chat"], latency="low")
        router = ModelRouter(FakeRegistry([slow, fast]), spend_tracker, mode="speed")

        decision = router.route(make_profile(task_type="simple_chat"))

        assert decision is not None
        assert decision.selected_model.name == "fast"
        assert "speed mode" in decision.reason


class TestScoringLogic:

    def test_local_model_preferred_when_tied(self, spend_tracker):
        """A free local model should win over cloud when strengths are equal."""
        local_model = make_model(name="local", local=True,  strengths=["simple_chat"], cost_per_1k_input=0.0)
        cloud_model = make_model(name="cloud", local=False, strengths=["simple_chat"], cost_per_1k_input=0.001)
        router = ModelRouter(FakeRegistry([local_model, cloud_model]), spend_tracker)

        decision = router.route(make_profile(task_type="simple_chat", difficulty="easy"))
        assert decision is not None
        assert decision.selected_model.name == "local"

    def test_strength_match_outweighs_local_bonus_for_hard_task(self, spend_tracker):
        """For a hard task, the model with the right strength wins over a generic local model."""
        specialist = make_model(name="specialist", local=False, strengths=["reasoning"],
                                cost_per_1k_input=0.003, api_key_env=None)
        generalist = make_model(name="local-general", local=True, strengths=["simple_chat"],
                                cost_per_1k_input=0.0)
        router = ModelRouter(FakeRegistry([specialist, generalist]), spend_tracker)

        decision = router.route(make_profile(task_type="reasoning", difficulty="hard"))
        assert decision is not None
        assert decision.selected_model.name == "specialist"

    def test_alternatives_list_populated(self, spend_tracker):
        models = [
            make_model(name=f"model-{i}", strengths=["coding"]) for i in range(4)
        ]
        router = ModelRouter(FakeRegistry(models), spend_tracker)
        decision = router.route(make_profile(task_type="coding"))
        assert decision is not None
        assert len(decision.alternatives) > 0

    def test_balanced_mode_uses_quality_floor_not_just_lowest_price(self, spend_tracker):
        weak_cheap = make_model(
            name="weak-cheap",
            strengths=["coding"],
            quality_score=0.50,
            task_quality={"coding": 0.50},
            cost_per_1k_input=0.00001,
            cost_per_1k_output=0.00002,
        )
        strong_mid = make_model(
            name="strong-mid",
            strengths=["coding"],
            quality_score=0.84,
            task_quality={"coding": 0.84},
            cost_per_1k_input=0.0004,
            cost_per_1k_output=0.0016,
        )
        router = ModelRouter(FakeRegistry([weak_cheap, strong_mid]), spend_tracker, mode="balanced")
        decision = router.route(make_profile(task_type="coding", difficulty="medium"))
        assert decision is not None
        assert decision.selected_model.name == "strong-mid"
        assert decision.quality_estimate >= 0.80

    def test_quality_mode_reports_best_quality_reference(self, spend_tracker):
        cheap = make_model(
            name="cheap-good",
            strengths=["summarization"],
            task_quality={"summarization": 0.76},
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
        )
        premium = make_model(
            name="premium-best",
            strengths=["summarization"],
            task_quality={"summarization": 0.92},
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )
        router = ModelRouter(FakeRegistry([cheap, premium]), spend_tracker, mode="quality")
        decision = router.route(make_profile(task_type="summarization", difficulty="hard", context_tokens=4000))
        assert decision is not None
        assert decision.best_quality_model == "premium-best"
        assert decision.premium_reference_cost_usd is not None
        assert decision.estimated_savings_usd is not None
