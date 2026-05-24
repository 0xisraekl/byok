"""Tests for BYOK doctor diagnostics."""

from pathlib import Path

from click.testing import CliRunner

from byok.cli.main import cli
from byok.core.doctor import DoctorCheck, check_config_file, check_model_keys
from byok.core.registry import ModelConfig


def test_check_config_file_reports_present(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("models: []\n")

    check = check_config_file(config)

    assert check.name == "Config file"
    assert check.status == "ok"
    assert str(config) in check.detail


def test_check_config_file_reports_missing(tmp_path):
    config = tmp_path / "missing.yaml"

    check = check_config_file(config)

    assert check.status == "error"
    assert "not found" in check.detail


def test_check_model_keys_summarizes_local_cloud_and_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    models = [
        ModelConfig(name="local", provider="ollama", model_id="llama", local=True),
        ModelConfig(name="openai", provider="openai", model_id="gpt", api_key_env="OPENAI_API_KEY"),
        ModelConfig(name="claude", provider="anthropic", model_id="claude", api_key_env="ANTHROPIC_API_KEY"),
        ModelConfig(name="off", provider="openai", model_id="gpt", api_key_env="OPENAI_API_KEY", enabled=False),
    ]

    check = check_model_keys(models)

    assert check.status == "warning"
    assert "2 ready" in check.detail
    assert "1 missing keys" in check.detail
    assert "1 disabled" in check.detail


def test_doctor_command_runs_and_prints_summary():
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "BYOK Doctor" in result.output
    assert "Config file" in result.output
    assert "Model keys" in result.output
