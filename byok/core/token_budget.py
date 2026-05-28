"""
token_budget.py — Output Token Budgeting

BYOK saves money not only by picking cheaper models, but also by avoiding
unbounded completions when a task likely needs a shorter answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from byok.core.classifier import TaskProfile


@dataclass
class TokenBudget:
    """Output-token cap recommended for one routed request."""

    max_output_tokens: int
    raw_estimated_output_tokens: int
    saved_tokens: int
    savings_pct: float
    reason: str


class TokenBudgeter:
    """Choose a practical output cap from task type, difficulty, and mode."""

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

    MODE_MULTIPLIERS = {
        "cheap": 0.60,
        "balanced": 0.80,
        "speed": 0.70,
        "private": 0.75,
        "quality": 1.00,
    }

    MIN_OUTPUT_TOKENS = {
        "simple_chat": 60,
        "extraction": 120,
        "summarization": 160,
        "writing": 220,
        "coding": 320,
        "reasoning": 240,
        "math": 180,
        "data_analysis": 320,
        "tool_calling": 120,
    }

    def budget_for(
        self,
        task: TaskProfile,
        mode: str = "balanced",
        requested_max_tokens: Optional[int] = None,
    ) -> TokenBudget:
        raw_estimate = self.estimate_output_tokens(task)
        multiplier = self.MODE_MULTIPLIERS.get(mode, self.MODE_MULTIPLIERS["balanced"])
        floor = self.MIN_OUTPUT_TOKENS.get(task.task_type, 120)
        recommended = max(floor, int(raw_estimate * multiplier))

        reason_parts = [f"{mode} mode token budget"]
        if requested_max_tokens is not None:
            recommended = min(recommended, requested_max_tokens)
            reason_parts.append("respects user max_tokens")

        saved = max(raw_estimate - recommended, 0)
        savings_pct = (saved / raw_estimate * 100) if raw_estimate > 0 else 0.0
        if saved > 0:
            reason_parts.append(f"caps output to save ~{savings_pct:.0f}% tokens")
        else:
            reason_parts.append("full output budget preserved")

        return TokenBudget(
            max_output_tokens=recommended,
            raw_estimated_output_tokens=raw_estimate,
            saved_tokens=saved,
            savings_pct=savings_pct,
            reason=", ".join(reason_parts),
        )

    def estimate_output_tokens(self, task: TaskProfile) -> int:
        by_difficulty = self.OUTPUT_TOKEN_ESTIMATES.get(
            task.task_type,
            self.OUTPUT_TOKEN_ESTIMATES["simple_chat"],
        )
        return by_difficulty.get(task.difficulty, by_difficulty["medium"])
