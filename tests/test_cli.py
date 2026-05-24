"""Tests for BYOK CLI route behavior."""

from click.testing import CliRunner

from byok.cli.main import cli


def test_route_command_accepts_mode_option():
    result = CliRunner().invoke(cli, ["route", "hi", "--mode", "private"])

    assert result.exit_code == 0
    assert "Mode" in result.output
    assert "private" in result.output
