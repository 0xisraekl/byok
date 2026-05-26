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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from byok.core.classifier import TaskClassifier
from byok.core.registry import ModelConfig, ModelRegistry
from byok.core.router import ModelRouter, RoutingDecision
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
                "-H 'Authorization: Bearer byok' "
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
    max_tokens: Optional[int] = body.get("max_tokens")
    requested_model: str = body.get("model", "auto")
    route_mode = _mode_from_request(requested_model, body.get("byok_mode"))

    if not messages:
        raise HTTPException(status_code=400, detail="messages field is required")

    # ── Classify the task ─────────────────────────────────────────────────
    task = classifier.classify(messages, tools)

    # ── Route to best model ───────────────────────────────────────────────
    request_router = ModelRouter(registry, spend_tracker, mode=route_mode)
    decision = request_router.route(task)

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

    # ── Call the selected provider ────────────────────────────────────────
    try:
        provider = _get_provider(model)
        chat_response = await provider.chat_completion(
            model_id=model.model_id,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        # If the chosen model fails, we could add fallback logic here in V2.
        raise HTTPException(
            status_code=502,
            detail=f"Provider error ({model.name}): {str(exc)}",
        )

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
        "byok": _last_decision["routing"] | {"cost_usd": round(cost, 6)},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Request helpers
# ─────────────────────────────────────────────────────────────────────────────

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
