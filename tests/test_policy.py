"""Tests for role-based routing policy."""

from pathlib import Path

from byok.core.policy import RoutingPolicy


def test_policy_loads_agent_defaults_from_yaml():
    policy = RoutingPolicy(Path("config/routing_policy.yaml"))

    controls = policy.controls_for("coding_agent")

    assert controls.mode == "balanced"
    assert controls.max_cost_usd == 0.004
    assert controls.max_output_tokens == 1200
    assert controls.source == "agent:coding_agent"
    assert policy.task_for_agent("coding_agent") == "coding"


def test_policy_supports_agent_aliases():
    policy = RoutingPolicy(Path("config/routing_policy.yaml"))

    controls = policy.controls_for("coder")

    assert controls.max_cost_usd == 0.004
    assert policy.task_for_agent("coder") == "coding"


def test_request_overrides_policy_defaults():
    policy = RoutingPolicy(Path("config/routing_policy.yaml"))

    controls = policy.controls_for(
        "writer",
        explicit_mode="quality",
        explicit_max_cost_usd=0.25,
        explicit_max_output_tokens=3000,
    )

    assert controls.mode == "quality"
    assert controls.max_cost_usd == 0.25
    assert controls.max_output_tokens == 3000
    assert controls.source == "agent:writer+request"
