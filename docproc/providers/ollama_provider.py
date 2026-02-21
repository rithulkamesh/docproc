"""Ollama provider for local models."""

import base64
import os
from typing import List, Optional

from docproc.providers.base import ChatMessage, ChatResponse, ModelProvider


class OllamaProvider(ModelProvider):
    """Ollama provider for local LLMs (LLaVA, Llama 3.2 Vision, etc.)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: str = "llava",
        default_vision_model: str = "llava",
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = default_model
        self.default_vision_model = default_vision_model

    def _to_ollama_messages(self, messages: List[ChatMessage]) -> list:
        out = []
        for m in messages:
            content = m.content
            if isinstance(content, list):
                # Ollama expects content as list of {"type": "text", "text": "..."} or {"type": "image", "image": base64}
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                b64 = url.split(",", 1)[-1]
                                parts.append({"type": "image", "image": b64})
                        elif block.get("type") == "text":
                            parts.append({"type": "text", "text": block.get("text", "")})
                    else:
                        parts.append({"type": "text", "text": str(block)})
                out.append({"role": m.role, "content": parts})
            else:
                out.append({"role": m.role, "content": str(content)})
        return out

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        import httpx
        model = model or self.default_model
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": self._to_ollama_messages(messages), "stream": False, **kwargs},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return ChatResponse(content=content, model=model, raw=data)

    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        return self.chat(messages, model=model or self.default_vision_model, **kwargs)

    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        import httpx
        model = model or "nomic-embed-text"
        embeddings = []
        for t in texts:
            resp = httpx.post(
                f"{self.base_url.rstrip('/')}/api/embeddings",
                json={"model": model, "prompt": t, **kwargs},
                timeout=60.0,
            )
            resp.raise_for_status()
            embeddings.append(resp.json().get("embedding", []))
        return embeddings
