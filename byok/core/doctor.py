"""doctor.py — BYOK environment diagnostics."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from byok.core.registry import ModelConfig


@dataclass(frozen=True)
class DoctorCheck:
    """One diagnostic result shown by `byok doctor`."""

    name: str
    status: str  # ok | warning | error
    detail: str


def check_config_file(config_path: str | Path) -> DoctorCheck:
    """Report whether the model config file exists."""
    path = Path(config_path)
    if path.exists():
        return DoctorCheck("Config file", "ok", f"Found {path}")
    return DoctorCheck("Config file", "error", f"Model config not found: {path}")


def check_model_keys(models: list[ModelConfig]) -> DoctorCheck:
    """Summarize enabled/disabled models and missing API keys."""
    disabled = [m for m in models if not m.enabled]
    enabled = [m for m in models if m.enabled]
    ready = [m for m in enabled if m.has_valid_key]
    missing = [m for m in enabled if not m.has_valid_key]

    detail = f"{len(ready)} ready, {len(missing)} missing keys, {len(disabled)} disabled"
    if missing:
        names = ", ".join(m.name for m in missing[:4])
        detail += f" ({names})"
        return DoctorCheck("Model keys", "warning", detail)
    if not ready:
        return DoctorCheck("Model keys", "error", detail)
    return DoctorCheck("Model keys", "ok", detail)


def check_spend_db(db_path: str | Path) -> DoctorCheck:
    """Report where spend/routing history will be stored."""
    path = Path(db_path)
    parent = path.parent if str(path.parent) else Path(".")
    if parent.exists():
        if path.exists():
            return DoctorCheck("Spend database", "ok", f"Found {path}")
        return DoctorCheck("Spend database", "ok", f"Will create {path} on first write")
    return DoctorCheck("Spend database", "error", f"Directory does not exist: {parent}")


def check_ollama(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.25) -> DoctorCheck:
    """Lightweight TCP health check for local Ollama."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return DoctorCheck("Ollama", "ok", f"Listening at {host}:{port}")
    except OSError:
        return DoctorCheck("Ollama", "warning", f"Not reachable at {host}:{port} (local models may be unavailable)")


def build_doctor_checks(config_path: str | Path, db_path: str | Path, models: list[ModelConfig] | None = None) -> list[DoctorCheck]:
    """Build all diagnostics that do not require paid provider calls."""
    checks = [check_config_file(config_path), check_spend_db(db_path)]
    if models is not None:
        checks.append(check_model_keys(models))
        if any(m.enabled and m.local for m in models):
            checks.append(check_ollama())
    return checks
