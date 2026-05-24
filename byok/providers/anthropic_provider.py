"""
anthropic_provider.py — Anthropic (Claude) Provider

Anthropic's API uses a different format from OpenAI.
This provider:
  1. Converts the incoming OpenAI-format messages to Anthropic format
  2. Calls the Anthropic Messages API
  3. Converts the response back to our standard ChatResponse

So the rest of BYOK never has to think about Anthropic's format.
"""

from __future__ import annotations

from typing import Optional

import httpx

from byok.providers.base import BaseProvider, ChatResponse

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """
    Calls the Anthropic Messages API and returns a normalized ChatResponse.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:

        # ── Convert OpenAI messages → Anthropic format ────────────────────
        # Anthropic separates the system prompt from the message list.
        system_prompt: Optional[str] = None
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic takes system as a top-level field, not a message
                system_prompt = content
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})
            # Skip "tool" role for now (V1 simplification)

        # Anthropic requires at least one message
        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "(no message)"}]

        payload: dict = {
            "model": model_id,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Anthropic temperature is 0.0–1.0 (same scale as OpenAI)
        if temperature != 1.0:
            payload["temperature"] = min(temperature, 1.0)

        # ── Convert OpenAI tools → Anthropic tools ────────────────────────
        if tools:
            payload["tools"] = self._convert_tools(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # ── Extract content from Anthropic response ───────────────────────
        # Anthropic returns a list of content blocks (text, tool_use, etc.)
        text_content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_content += block["text"]
            elif block["type"] == "tool_use":
                # Convert to OpenAI tool_calls format so the rest of BYOK
                # and the agent framework can handle it uniformly
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": str(block.get("input", {})),
                    },
                })

        usage = data.get("usage", {})

        return ChatResponse(
            content=text_content,
            model_used=data.get("model", model_id),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            tool_calls=tool_calls if tool_calls else None,
            raw_response=data,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _convert_tools(self, openai_tools: list) -> list:
        """
        OpenAI tools format:
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Anthropic tools format:
          {"name": ..., "description": ..., "input_schema": {...}}
        """
        result = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                result.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                })
        return result
