"""Tests for BYOK token budget behavior."""

from byok.core.classifier import TaskProfile
from byok.core.token_budget import TokenBudgeter


def make_profile(**overrides) -> TaskProfile:
    defaults = dict(
        task_type="writing",
        secondary_types=[],
        difficulty="medium",
        context_tokens=500,
        has_tools=False,
        privacy_required=False,
        confidence=0.9,
    )
    defaults.update(overrides)
    return TaskProfile(**defaults)


def test_balanced_mode_caps_output_below_raw_estimate_to_save_tokens():
    budget = TokenBudgeter().budget_for(make_profile(task_type="writing", difficulty="medium"), mode="balanced")

    assert budget.raw_estimated_output_tokens == 900
    assert budget.max_output_tokens < budget.raw_estimated_output_tokens
    assert budget.max_output_tokens == 720
    assert "balanced" in budget.reason


def test_quality_mode_preserves_more_output_budget_than_cheap_mode():
    profile = make_profile(task_type="coding", difficulty="hard")

    cheap_budget = TokenBudgeter().budget_for(profile, mode="cheap")
    quality_budget = TokenBudgeter().budget_for(profile, mode="quality")

    assert cheap_budget.max_output_tokens < quality_budget.max_output_tokens
    assert quality_budget.max_output_tokens == 2800


def test_explicit_user_max_tokens_is_never_increased():
    budget = TokenBudgeter().budget_for(
        make_profile(task_type="coding", difficulty="hard"),
        mode="quality",
        requested_max_tokens=600,
    )

    assert budget.max_output_tokens == 600
    assert budget.saved_tokens > 0


def test_easy_simple_chat_gets_tight_default_budget():
    budget = TokenBudgeter().budget_for(make_profile(task_type="simple_chat", difficulty="easy"), mode="cheap")

    assert budget.max_output_tokens <= 100
    assert budget.saved_tokens > 0


def test_cost_ceiling_caps_output_tokens_for_paid_model():
    budget = TokenBudgeter().budget_for_model_cost(
        make_profile(task_type="coding", difficulty="medium", context_tokens=1000),
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.01,
        mode="quality",
        max_cost_usd=0.006,
    )

    # $0.001 input leaves $0.005 for output at $0.01 / 1k = 500 tokens.
    assert budget.max_output_tokens == 500
    assert "request cost ceiling" in budget.reason


def test_cost_ceiling_returns_zero_when_input_already_exceeds_budget():
    budget = TokenBudgeter().budget_for_model_cost(
        make_profile(task_type="summarization", difficulty="medium", context_tokens=10000),
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        mode="balanced",
        max_cost_usd=0.001,
    )

    assert budget.max_output_tokens == 0
    assert budget.savings_pct == 100.0
