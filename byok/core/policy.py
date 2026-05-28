"""
policy.py - Routing policy defaults for sub-agent roles.

The classifier can infer a role such as "coding_agent" from a system prompt.
This module maps that role to practical defaults: task type, routing mode,
per-call cost ceiling, and output-token cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


ALLOWED_MODES = {"balanced", "cheap", "quality", "private", "speed"}

AGENT_ALIASES = {
    "coder": "coding_agent",
    "coding": "coding_agent",
    "developer": "coding_agent",
    "software_engineer": "coding_agent",
    "research": "research_agent",
    "researcher": "research_agent",
    "analyst": "research_agent",
    "data": "data_agent",
    "data_analyst": "data_agent",
    "writer": "writing_agent",
    "editor": "writing_agent",
    "summarizer": "summarization_agent",
    "summary": "summarization_agent",
    "solver": "math_agent",
    "math": "math_agent",
    "tool": "tool_agent",
    "tools": "tool_agent",
    "browser": "tool_agent",
    "search": "tool_agent",
}


@dataclass
class RouteControls:
    """Resolved routing controls for one request."""

    mode: str = "balanced"
    max_cost_usd: Optional[float] = None
    max_output_tokens: Optional[int] = None
    source: str = "default"


class RoutingPolicy:
    """Load role defaults from config/routing_policy.yaml."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.defaults: dict[str, Any] = {"mode": "balanced"}
        self.agents: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.config_path.exists():
            return

        with open(self.config_path) as f:
            data = yaml.safe_load(f) or {}

        self.defaults = data.get("defaults", {}) or {"mode": "balanced"}
        self.agents = data.get("agents", {}) or {}

    def for_agent(self, agent_role: Optional[str]) -> dict[str, Any]:
        if not agent_role:
            return {}
        normalized = str(agent_role).strip().lower().replace("-", "_").replace(" ", "_")
        canonical = AGENT_ALIASES.get(normalized, normalized)
        return dict(self.agents.get(canonical, {}))

    def controls_for(
        self,
        agent_role: Optional[str],
        explicit_mode: Optional[str] = None,
        explicit_max_cost_usd: Optional[float] = None,
        explicit_max_output_tokens: Optional[int] = None,
    ) -> RouteControls:
        policy = self.for_agent(agent_role)
        source = f"agent:{agent_role}" if policy else "default"

        mode = explicit_mode or policy.get("mode") or self.defaults.get("mode") or "balanced"
        if mode not in ALLOWED_MODES:
            mode = "balanced"

        max_cost = explicit_max_cost_usd
        if max_cost is None:
            max_cost = _optional_float(policy.get("max_cost_usd"))

        max_output = explicit_max_output_tokens
        if max_output is None:
            max_output = _optional_int(policy.get("max_output_tokens"))

        if explicit_mode or explicit_max_cost_usd is not None or explicit_max_output_tokens is not None:
            source = f"{source}+request"

        return RouteControls(
            mode=mode,
            max_cost_usd=max_cost,
            max_output_tokens=max_output,
            source=source,
        )

    def task_for_agent(self, agent_role: Optional[str]) -> Optional[str]:
        value = self.for_agent(agent_role).get("task")
        return str(value) if value else None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
