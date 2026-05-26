"""
registry.py — Model Registry

Loads your model pool from config/models.yaml and answers questions like:
  - Which models support tool calling?
  - Which models are under their spend limit?
  - Which models are good at coding?

This is the "source of truth" for what BYOK is allowed to route to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """All the information BYOK needs to know about one model."""

    # Identity
    name: str           # Your label, e.g. "my-claude" — shown in logs
    provider: str       # openai | anthropic | ollama | openai_compatible
    model_id: str       # The actual model name sent to the API

    # What it's good at
    strengths: list[str] = field(default_factory=list)

    # Capabilities
    context_window: int = 8192
    supports_tools: bool = False
    local: bool = False

    # Cost (USD per 1,000 tokens)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Performance
    latency: str = "medium"   # low | medium | high
    quality_score: float = 0.70  # 0.0-1.0 general quality prior
    task_quality: dict[str, float] = field(default_factory=dict)  # optional per-task quality priors

    # Connection
    api_key_env: Optional[str] = None   # name of the env var holding the key
    base_url: Optional[str] = None      # for openai_compatible providers

    # Budget control
    spend_limit_monthly_usd: float = 0.0  # 0 = unlimited
    enabled: bool = True

    # Routing priority (1 = highest, used as tiebreaker)
    priority: int = 2

    @property
    def api_key(self) -> Optional[str]:
        """Read the API key from the environment at runtime."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    @property
    def has_valid_key(self) -> bool:
        """True if this model has a usable API key (or doesn't need one)."""
        if self.local:
            return True  # Ollama needs no key
        return bool(self.api_key)

    def can_handle_tokens(self, token_count: int) -> bool:
        """True if this model's context window fits the request."""
        return self.context_window >= token_count

    def __repr__(self) -> str:
        return f"ModelConfig(name={self.name!r}, provider={self.provider!r})"


class ModelRegistry:
    """
    Loads models.yaml and gives the router a clean interface to query models.

    Usage:
        registry = ModelRegistry("config/models.yaml")
        models = registry.get_capable_models(
            task_type="coding",
            needs_tools=True,
            min_context=4000,
        )
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._models: list[ModelConfig] = []
        self.load()

    def load(self) -> None:
        """Read models.yaml and populate the registry."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {self.config_path}\n"
                f"Expected a models.yaml file at that path."
            )

        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        self._models = []
        for entry in data.get("models", []):
            model = ModelConfig(
                name=entry["name"],
                provider=entry["provider"],
                model_id=entry["model_id"],
                strengths=entry.get("strengths", []),
                context_window=entry.get("context_window", 8192),
                supports_tools=entry.get("supports_tools", False),
                local=entry.get("local", False),
                cost_per_1k_input=entry.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=entry.get("cost_per_1k_output", 0.0),
                latency=entry.get("latency", "medium"),
                quality_score=entry.get("quality_score", 0.70),
                task_quality=entry.get("task_quality", {}),
                api_key_env=entry.get("api_key_env"),
                base_url=entry.get("base_url"),
                spend_limit_monthly_usd=entry.get("spend_limit_monthly_usd", 0.0),
                enabled=entry.get("enabled", True),
                priority=entry.get("priority", 2),
            )
            self._models.append(model)

    def all_models(self) -> list[ModelConfig]:
        """Return every model in the registry (including disabled ones)."""
        return list(self._models)

    def available_models(self) -> list[ModelConfig]:
        """Return only models that are enabled AND have a valid API key."""
        return [m for m in self._models if m.enabled and m.has_valid_key]

    def get(self, name: str) -> Optional[ModelConfig]:
        """Look up a model by its name label."""
        for m in self._models:
            if m.name == name:
                return m
        return None

    def get_capable_models(
        self,
        task_type: str,
        needs_tools: bool = False,
        min_context: int = 0,
        local_only: bool = False,
    ) -> list[ModelConfig]:
        """
        Return all models that CAN handle a given task.
        This is the first filter — a model must pass ALL conditions to be included.
        The router then scores and ranks the results.
        """
        candidates = []
        for model in self.available_models():
            # Must support the task type (or be a general model)
            if task_type not in model.strengths and "general" not in model.strengths:
                continue

            # Must support tools if the task needs them
            if needs_tools and not model.supports_tools:
                continue

            # Must have a large enough context window
            if not model.can_handle_tokens(min_context):
                continue

            # Must be local if privacy mode is on
            if local_only and not model.local:
                continue

            candidates.append(model)

        return candidates

    def summary(self) -> str:
        """Return a human-readable summary for CLI display."""
        lines = [f"  {'NAME':<22} {'PROVIDER':<16} {'STRENGTHS':<40} {'LIMIT'}"]
        lines.append("  " + "─" * 90)
        for m in self._models:
            status = "✓" if (m.enabled and m.has_valid_key) else "✗"
            limit = f"${m.spend_limit_monthly_usd:.0f}/mo" if m.spend_limit_monthly_usd > 0 else "unlimited"
            strengths = ", ".join(m.strengths[:3])
            lines.append(
                f"  {status} {m.name:<20} {m.provider:<16} {strengths:<40} {limit}"
            )
        return "\n".join(lines)
