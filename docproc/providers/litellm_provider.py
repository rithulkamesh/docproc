"""LiteLLM unified provider — routes to OpenAI, Anthropic, Azure, Ollama, etc."""

import base64
import logging
from typing import List, Optional

from docproc.providers.base import ChatMessage, ChatResponse, ModelProvider

logger = logging.getLogger(__name__)


class LiteLLMProvider(ModelProvider):
    """Unified AI provider via LiteLLM. Single interface for many backends."""

    def __init__(
        self,
        default_model: str = "gpt-4o",
        default_vision_model: str = "gpt-4o",
    ):
        """Initialize LiteLLM provider.

        Uses LiteLLM model strings: openai/gpt-4o, anthropic/claude-3-5-sonnet,
        azure/gpt-4o, ollama/llava, etc.
        """
        self.default_model = default_model
        self.default_vision_model = default_vision_model

    def _to_litellm_messages(self, messages: List[ChatMessage]) -> list:
        """Convert ChatMessage list to LiteLLM format."""
        out = []
        for m in messages:
            content = m.content
            if isinstance(content, list):
                out.append({"role": m.role, "content": content})
            else:
                out.append({"role": m.role, "content": str(content)})
        return out

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        import litellm

        model = model or self.default_model
        msgs = self._to_litellm_messages(messages)
        resp = litellm.completion(model=model, messages=msgs, **kwargs)
        content = resp.choices[0].message.content or ""
        usage = None
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return ChatResponse(content=content, model=resp.model, usage=usage, raw=resp)

    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        return self.chat(messages, model=model or self.default_vision_model, **kwargs)

    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        import litellm

        model = model or "text-embedding-3-small"
        resp = litellm.embedding(model=model, input=texts, **kwargs)
        return [d["embedding"] for d in resp.data]
