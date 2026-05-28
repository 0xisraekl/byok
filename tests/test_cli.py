"""Tests for BYOK CLI route behavior."""

from click.testing import CliRunner

from byok.cli.main import cli


def test_route_command_accepts_mode_option():
    result = CliRunner().invoke(cli, ["route", "hi", "--mode", "private"])

    assert result.exit_code == 0
    assert "Mode" in result.output
    assert "private" in result.output


def test_route_command_shows_token_budget():
    result = CliRunner().invoke(cli, ["route", "Draft a quick email", "--mode", "cheap"])

    assert result.exit_code == 0
    assert "Token budget" in result.output
    assert "Max output tokens" in result.output


def test_route_command_accepts_request_cost_and_output_limits():
    result = CliRunner().invoke(
        cli,
        [
            "route",
            "Write a Python function",
            "--max-cost",
            "0.001",
            "--max-output-tokens",
            "200",
        ],
    )

    assert result.exit_code == 0
    assert "Max cost" in result.output
    assert "$0.00100" in result.output
    assert "Token budget" in result.output


def test_specialties_command_shows_best_models_by_task():
    result = CliRunner().invoke(cli, ["specialties"])

    assert result.exit_code == 0
    assert "Best models by task" in result.output
    assert "coding" in result.output
    assert "Best quality" in result.output
