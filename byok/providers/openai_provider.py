"""
openai_provider.py — OpenAI Provider

Handles any provider that speaks the OpenAI API format:
  - OpenAI (api.openai.com)
  - DeepSeek (api.deepseek.com)
  - Groq (api.groq.com)
  - Any OpenAI-compatible local server

Set base_url to point to different endpoints.
"""

from __future__ import annotations

from typing import Optional

import httpx

from byok.providers.base import BaseProvider, ChatResponse


class OpenAIProvider(BaseProvider):
    """
    Calls the OpenAI Chat Completions API (or any compatible endpoint).
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Allow any extra kwargs to be passed through (stream, etc.)
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")

        return ChatResponse(
            content=message.get("content") or "",
            model_used=data.get("model", model_id),
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            tool_calls=tool_calls,
            raw_response=data,
        )
