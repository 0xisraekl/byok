"""
ollama_provider.py — Ollama Provider (Local Models)

Ollama runs entirely on your machine — free, private, no API key needed.
Install Ollama: https://ollama.com
Pull a model:   ollama pull gemma3:6b

Ollama's API is OpenAI-compatible, so this is essentially the OpenAI
provider pointing at localhost — but we keep it separate so it has
clear local-specific error messages and configuration.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from byok.providers.base import BaseProvider, ChatResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """
    Calls a locally running Ollama instance.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:

        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )

        message = data.get("message", {})
        content = message.get("content", "")

        # Ollama doesn't return token counts in all versions,
        # so we fall back to estimates
        usage = data.get("usage", {})
        input_tokens = (
            usage.get("prompt_tokens")
            or data.get("prompt_eval_count")
            or len(" ".join(m.get("content", "") for m in messages)) // 4
        )
        output_tokens = (
            usage.get("completion_tokens")
            or data.get("eval_count")
            or len(content) // 4
        )

        return ChatResponse(
            content=content,
            model_used=data.get("model", model_id),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=data,
        )
