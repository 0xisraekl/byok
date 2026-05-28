"""
server.py — BYOK Proxy Server

This is an OpenAI-compatible API server.
Hermes Agent (and any other framework) can point their base_url here
and they'll automatically get intelligent model routing.

Hermes integration (just change one config value):
    base_url = "http://localhost:8000/v1"
    api_key  = "byok"          ← any non-empty string

Endpoints:
    POST /v1/chat/completions  ← main routing endpoint
    GET  /v1/models            ← returns your configured model pool
    GET  /v1/routing/last      ← see the last routing decision (debug)
    GET  /health               ← health check
    GET  /                     ← welcome page with setup info
"""

from __future__ import annotations

import time
import uuid
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from byok.core.classifier import TASK_SIGNALS, TaskClassifier, TaskProfile
from byok.core.registry import ModelConfig, ModelRegistry
from byok.core.router import ModelRouter, RoutingDecision
from byok.core.token_budget import TokenBudgeter
from byok.providers.anthropic_provider import AnthropicProvider
from byok.providers.ollama_provider import OllamaProvider
from byok.providers.openai_provider import OpenAIProvider
from byok.storage.spend_tracker import SpendTracker

# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap: load config, create shared instances
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"
DB_PATH = Path("byok.db")

registry = ModelRegistry(CONFIG_PATH)
spend_tracker = SpendTracker(DB_PATH)
classifier = TaskClassifier()
router = ModelRouter(registry, spend_tracker)
token_budgeter = TokenBudgeter()

# Holds the last routing decision so /v1/routing/last can return it
_last_decision: Optional[dict] = None

# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BYOK — Bring Your Own Key Router",
    description=(
        "An OpenAI-compatible proxy that routes each request to the best model "
        "in your configured pool. Point your agent's base_url here."
    ),
    version="0.1.0",
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Welcome page with quick-start instructions."""
    available = registry.available_models()
    return {
        "service": "BYOK Router",
        "version": "0.1.0",
        "status": "running",
        "models_in_pool": len(available),
        "model_names": [m.name for m in available],
        "quick_start": {
            "hermes_agent": "Set base_url='http://localhost:8000/v1' and api_key='byok'",
            "test_with_curl": (
                "curl http://localhost:8000/v1/chat/completions "
                "-H 'Authorization: Bearer *** "
                "-H 'Content-Type: application/json' "
                "-d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}'"
            ),
            "docs": "http://localhost:8000/docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/v1/models")
async def list_models():
    """
    Return available models in OpenAI format.
    Agent frameworks call this to discover what models are available.
    """
    models = registry.available_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": 1700000000,
                "owned_by": f"byok/{m.provider}",
            }
            for m in models
        ],
    }


@app.get("/v1/routing/last")
async def last_routing_decision():
    """
    Return the most recent routing decision.
    Useful for debugging: see what BYOK chose and why.
    """
    if _last_decision is None:
        return {"message": "No requests have been routed yet."}
    return _last_decision


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Main routing endpoint.

    1. Parse the request (OpenAI format)
    2. Classify the task
    3. Route to the best model in your pool
    4. Call that model's API
    5. Log the decision + cost
    6. Return an OpenAI-format response
    """
    global _last_decision

    # ── Parse request ─────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    messages: list[dict] = body.get("messages", [])
    tools: list = body.get("tools", [])
    temperature: float = body.get("temperature", 0.7)
    byok_meta: dict[str, Any] = body.get("byok") if isinstance(body.get("byok"), dict) else {}
    max_tokens: Optional[int] = (
        body.get("max_tokens")
        or body.get("max_completion_tokens")
        or byok_meta.get("max_output_tokens")
        or byok_meta.get("max_tokens")
    )
    max_cost_usd = _optional_float(
        byok_meta.get("max_cost_usd")
        or byok_meta.get("budget_usd")
        or body.get("byok_max_cost_usd")
    )
    requested_model: str = body.get("model", "auto")
    route_mode = _mode_from_request(requested_model, body.get("byok_mode") or byok_meta.get("mode"))

    if not messages:
        raise HTTPException(status_code=400, detail="messages field is required")

    # ── Classify the task ─────────────────────────────────────────────────
    task = classifier.classify(messages, tools)
    task = _apply_byok_metadata(task, byok_meta)
    provider_messages = _strip_byok_hints_from_messages(messages)

    # ── Route to best model ───────────────────────────────────────────────
    request_router = ModelRouter(registry, spend_tracker, mode=route_mode)
    decision = request_router.route(task, max_cost_usd=max_cost_usd, requested_max_tokens=max_tokens)

    if decision is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model available for this task. "
                "Check your models.yaml — all models may be disabled, "
                "at their spend limit, or missing API keys."
            ),
        )

    model = decision.selected_model
    token_budget = token_budgeter.budget_for_model_cost(
        task=task,
        cost_per_1k_input=model.cost_per_1k_input,
        cost_per_1k_output=model.cost_per_1k_output,
        mode=route_mode,
        requested_max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
    )
    max_tokens = token_budget.max_output_tokens

    # ── Call the selected provider, with transparent fallback ──────────────
    # The router's first choice is still the product decision. If that
    # provider is temporarily down/rate-limited, try the ranked runners-up
    # instead of failing the whole request. This is a big practical reliability
    # win for BYOK because users often bring multiple keys/providers.
    attempted_models: list[str] = []
    fallback_errors: list[dict[str, str]] = []
    chat_response = None
    selected_by_router = model.name

    for attempt_model in _attempt_models(decision, registry):
        attempted_models.append(attempt_model.name)
        try:
            provider = _get_provider(attempt_model)
            chat_response = await provider.chat_completion(
                model_id=attempt_model.model_id,
                messages=provider_messages,
                tools=tools if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            model = attempt_model
            break
        except Exception as exc:
            fallback_errors.append({"model": attempt_model.name, "error": str(exc)})

    if chat_response is None:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All routed provider attempts failed.",
                "attempted_models": attempted_models,
                "errors": fallback_errors,
            },
        )

    fallback_from = selected_by_router if model.name != selected_by_router else None

    # ── Calculate cost and log ────────────────────────────────────────────
    cost = (
        chat_response.input_tokens * model.cost_per_1k_input / 1000
        + chat_response.output_tokens * model.cost_per_1k_output / 1000
    )

    spend_tracker.log(
        model_name=model.name,
        provider=model.provider,
        task_type=task.task_type,
        difficulty=task.difficulty,
        input_tokens=chat_response.input_tokens,
        output_tokens=chat_response.output_tokens,
        cost_usd=cost,
        routing_reason=decision.reason,
    )

    # ── Store last decision for /v1/routing/last ──────────────────────────
    _last_decision = {
        "task": {
            "type": task.task_type,
            "difficulty": task.difficulty,
            "context_tokens": task.context_tokens,
            "has_tools": task.has_tools,
            "privacy_required": task.privacy_required,
            "confidence": round(task.confidence, 2),
            "agent_role": task.agent_role,
            "route_hints": task.route_hints,
        },
        "routing": {
            "mode": route_mode,
            "selected_model": model.name,
            "provider": model.provider,
            "reason": decision.reason,
            "score": round(decision.score, 2),
            "quality_estimate": round(decision.quality_estimate, 2),
            "best_quality_model": decision.best_quality_model,
            "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
            "premium_reference_cost_usd": round(decision.premium_reference_cost_usd, 6) if decision.premium_reference_cost_usd is not None else None,
            "estimated_savings_usd": round(decision.estimated_savings_usd, 6) if decision.estimated_savings_usd is not None else None,
            "estimated_savings_pct": round(decision.estimated_savings_pct, 1) if decision.estimated_savings_pct is not None else None,
            "alternatives": decision.alternatives,
            "attempted_models": attempted_models,
            "fallback_from": fallback_from,
            "fallback_errors": fallback_errors if fallback_from else [],
        },
        "token_budget": {
            "max_output_tokens": token_budget.max_output_tokens,
            "raw_estimated_output_tokens": token_budget.raw_estimated_output_tokens,
            "saved_tokens": token_budget.saved_tokens,
            "savings_pct": round(token_budget.savings_pct, 1),
            "reason": token_budget.reason,
            "request_max_cost_usd": max_cost_usd,
        },
        "usage": {
            "input_tokens": chat_response.input_tokens,
            "output_tokens": chat_response.output_tokens,
            "cost_usd": round(cost, 6),
        },
    }

    # ── Build OpenAI-format response ──────────────────────────────────────
    response_message: dict[str, Any] = {
        "role": "assistant",
        "content": chat_response.content,
    }

    # Pass tool calls through if the model made any
    if chat_response.tool_calls:
        response_message["tool_calls"] = chat_response.tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model.name,           # show BYOK model name, not raw API model
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": "tool_calls" if chat_response.tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": chat_response.input_tokens,
            "completion_tokens": chat_response.output_tokens,
            "total_tokens": chat_response.input_tokens + chat_response.output_tokens,
        },
        # BYOK metadata — extra field your agent can optionally read
        "byok": _last_decision["routing"] | {
            "cost_usd": round(cost, 6),
            "token_budget": _last_decision["token_budget"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Request helpers
# ─────────────────────────────────────────────────────────────────────────────


class ModelLookup(Protocol):
    def get(self, name: str) -> Optional[ModelConfig]: ...


def _attempt_models(decision: RoutingDecision, model_registry: ModelLookup) -> list[ModelConfig]:
    """Return selected model followed by ranked fallback candidates.

    `RoutingDecision.alternatives` intentionally stores lightweight
    `(model_name, score)` tuples for logs/API responses. The proxy expands those
    names back to configs when it needs to retry after a provider failure.
    """
    attempts: list[ModelConfig] = [decision.selected_model]
    seen = {decision.selected_model.name}

    for model_name, _score in decision.alternatives:
        if model_name in seen:
            continue
        model = model_registry.get(model_name)
        if model is None:
            continue
        attempts.append(model)
        seen.add(model.name)

    return attempts


def _mode_from_request(requested_model: str, explicit_mode: Optional[str]) -> str:
    """Support OpenAI-compatible model names like auto:cheap / auto:quality."""
    allowed = {"balanced", "cheap", "quality", "private", "speed"}
    if explicit_mode in allowed:
        return explicit_mode
    if isinstance(requested_model, str) and requested_model.startswith("auto:"):
        _, _, mode = requested_model.partition(":")
        if mode in allowed:
            return mode
    return "balanced"


def _apply_byok_metadata(task: TaskProfile, metadata: dict[str, Any]) -> TaskProfile:
    """
    Apply explicit request metadata after prompt-based classification.

    This is how Hermes/OpenClaw users can route sub-agent calls without
    modifying BYOK internals:

        {
          "model": "auto",
          "messages": [...],
          "byok": {"task": "coding", "agent": "coder", "privacy": true}
        }
    """
    if not metadata:
        return task

    task_type = metadata.get("task") or metadata.get("task_type")
    if task_type not in TASK_SIGNALS:
        task_type = task.task_type

    difficulty = metadata.get("difficulty", task.difficulty)
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = task.difficulty

    privacy_value = metadata.get("privacy", metadata.get("private", metadata.get("local_only", False)))
    privacy_required = task.privacy_required or _truthy(privacy_value)

    role = metadata.get("agent") or metadata.get("agent_role") or task.agent_role
    route_hints = dict(task.route_hints)
    for key in ("task", "task_type", "agent", "agent_role", "mode", "privacy", "private", "local_only"):
        if key in metadata:
            route_hints[key] = str(metadata[key]).lower()

    return replace(
        task,
        task_type=task_type,
        difficulty=difficulty,
        privacy_required=privacy_required,
        agent_role=role,
        route_hints=route_hints,
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "local"}


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _strip_byok_hints_from_messages(messages: list[dict]) -> list[dict]:
    """Remove inline [byok:*] routing hints before sending prompts to providers."""
    cleaned: list[dict] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, str):
            copied["content"] = _strip_byok_hints(content)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    part_copy = dict(part)
                    if part_copy.get("type") == "text" and isinstance(part_copy.get("text"), str):
                        part_copy["text"] = _strip_byok_hints(part_copy["text"])
                    parts.append(part_copy)
                else:
                    parts.append(part)
            copied["content"] = parts
        cleaned.append(copied)
    return cleaned


def _strip_byok_hints(text: str) -> str:
    return re.sub(r"\s*\[byok:[^\]]+\]\s*", " ", text, flags=re.IGNORECASE).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Provider factory — creates the right provider object for each model
# ─────────────────────────────────────────────────────────────────────────────

def _get_provider(model: ModelConfig):
    """Return the correct provider instance for a given model config."""

    if model.provider == "ollama":
        return OllamaProvider(base_url=model.base_url)

    if model.provider == "anthropic":
        api_key = model.api_key
        if not api_key:
            raise ValueError(
                f"No API key for {model.name}. "
                f"Set {model.api_key_env} in your .env file."
            )
        return AnthropicProvider(api_key=api_key)

    if model.provider in ("openai", "openai_compatible"):
        api_key = model.api_key
        if not api_key:
            raise ValueError(
                f"No API key for {model.name}. "
                f"Set {model.api_key_env} in your .env file."
            )
        base_url = model.base_url or "https://api.openai.com/v1"
        return OpenAIProvider(api_key=api_key, base_url=base_url)

    raise ValueError(
        f"Unknown provider '{model.provider}' for model '{model.name}'. "
        "Supported: openai, anthropic, ollama, openai_compatible"
    )
