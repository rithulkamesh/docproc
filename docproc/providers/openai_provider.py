"""OpenAI and Azure OpenAI provider."""

import base64
import os
from typing import List, Optional

from docproc.providers.base import ChatMessage, ChatResponse, ModelProvider


class OpenAIProvider(ModelProvider):
    """OpenAI API provider (also supports Azure via env vars)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o",
        default_vision_model: str = "gpt-4o",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
        self.default_model = default_model
        self.default_vision_model = default_vision_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _to_openai_messages(self, messages: List[ChatMessage]) -> list:
        out = []
        for m in messages:
            content = m.content
            if isinstance(content, list):
                out.append({"role": m.role, "content": content})
            elif isinstance(content, str):
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
        client = self._get_client()
        model = model or self.default_model
        resp = client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else None
        return ChatResponse(content=content, model=resp.model, usage=usage, raw=resp)

    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        # Vision uses same chat API with image content blocks
        return self.chat(messages, model=model or self.default_vision_model, **kwargs)

    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        client = self._get_client()
        model = model or "text-embedding-3-small"
        resp = client.embeddings.create(input=texts, model=model, **kwargs)
        return [d.embedding for d in resp.data]
