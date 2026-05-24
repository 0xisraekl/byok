"""
Tests for the spend tracker.
Run with:  .venv/bin/pytest tests/ -v
"""

import pytest
from byok.storage.spend_tracker import SpendTracker


@pytest.fixture
def tracker():
    return SpendTracker(":memory:")


class TestSpendTracking:

    def test_starts_at_zero(self, tracker):
        assert tracker.get_monthly_spend("any-model") == 0.0

    def test_logs_and_reads_spend(self, tracker):
        tracker.log("gpt-4o", "openai", "coding", "hard",
                    input_tokens=500, output_tokens=200, cost_usd=0.0042)
        assert tracker.get_monthly_spend("gpt-4o") == pytest.approx(0.0042)

    def test_multiple_requests_accumulate(self, tracker):
        tracker.log("claude", "anthropic", "writing", "medium",
                    input_tokens=100, output_tokens=100, cost_usd=0.001)
        tracker.log("claude", "anthropic", "writing", "medium",
                    input_tokens=100, output_tokens=100, cost_usd=0.002)
        assert tracker.get_monthly_spend("claude") == pytest.approx(0.003)

    def test_different_models_tracked_separately(self, tracker):
        tracker.log("model-a", "openai", "coding", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=0.005)
        tracker.log("model-b", "anthropic", "writing", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=0.010)
        assert tracker.get_monthly_spend("model-a") == pytest.approx(0.005)
        assert tracker.get_monthly_spend("model-b") == pytest.approx(0.010)

    def test_total_spent_sums_all_models(self, tracker):
        tracker.log("a", "openai", "coding", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=1.00)
        tracker.log("b", "anthropic", "writing", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=2.00)
        assert tracker.total_spent() == pytest.approx(3.00)

    def test_total_requests_counts(self, tracker):
        assert tracker.total_requests() == 0
        tracker.log("a", "openai", "coding", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.log("b", "anthropic", "coding", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=0.01)
        assert tracker.total_requests() == 2

    def test_get_recent_returns_latest_first(self, tracker):
        for i in range(5):
            tracker.log("model", "openai", "coding", "easy",
                        input_tokens=i * 10, output_tokens=10,
                        cost_usd=float(i) * 0.001)
        records = tracker.get_recent(3)
        assert len(records) == 3
        # Most recent should have the highest input tokens
        assert records[0].input_tokens > records[-1].input_tokens

    def test_get_all_monthly_spend(self, tracker):
        tracker.log("a", "openai", "coding", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=1.00)
        tracker.log("b", "anthropic", "writing", "easy",
                    input_tokens=100, output_tokens=50, cost_usd=2.50)
        monthly = tracker.get_all_monthly_spend()
        assert "a" in monthly
        assert "b" in monthly
        assert monthly["a"] == pytest.approx(1.00)
        assert monthly["b"] == pytest.approx(2.50)
