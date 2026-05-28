"""Tests for proxy fallback helpers."""

from byok.core.registry import ModelConfig
from byok.core.router import RoutingDecision
from byok.proxy.server import _attempt_models


def make_model(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="openai",
        model_id=name,
        strengths=["general"],
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
