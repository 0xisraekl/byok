"""
base.py — Provider Base Class

Every provider (OpenAI, Anthropic, Ollama, etc.) implements this interface.
The proxy server only ever talks to BaseProvider — it doesn't care which
real API is on the other end. All responses come back in the same shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatResponse:
    """
    Normalized response from any provider.
    Always in the same shape regardless of which API was called.
    """
    content: str                # the text the model returned
    model_used: str             # the actual model ID the API confirmed
    input_tokens: int           # tokens in the prompt
    output_tokens: int          # tokens in the response
    raw_response: dict          # the full original API response (for debugging)
    tool_calls: Optional[list] = None  # if the model called any tools


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    To add a new provider:
    1. Create a new file in byok/providers/
    2. Subclass BaseProvider
    3. Implement chat_completion()
    4. Register it in the ProviderFactory below
    """

    @abstractmethod
    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict],
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request to the provider.

        Args:
            model_id:    The model name as the API expects it
            messages:    OpenAI-format message list
            tools:       OpenAI-format tools array (if any)
            temperature: Sampling temperature
            max_tokens:  Max tokens to generate

        Returns:
            A normalized ChatResponse
        """
        ...
