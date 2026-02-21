"""Anthropic provider."""

import base64
import os
from typing import List, Optional

from docproc.providers.base import ChatMessage, ChatResponse, ModelProvider


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
        default_vision_model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = default_model
        self.default_vision_model = default_vision_model

    def _to_anthropic_content(self, content) -> list:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            out = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:image/png;base64,...
                            parts = url.split(",", 1)
                            if len(parts) == 2:
                                b64 = parts[1]
                                media_type = "image/png"
                                if ";" in parts[0]:
                                    media_type = parts[0].split(";")[0].replace("data:", "")
                                out.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
                    elif block.get("type") == "text":
                        out.append(block)
                else:
                    out.append({"type": "text", "text": str(block)})
            return out if out else [{"type": "text", "text": ""}]
        return [{"type": "text", "text": str(content)}]

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        model = model or self.default_model
        system = None
        msgs = []
        for m in messages:
            if m.role == "system":
                system = self._to_anthropic_content(m.content)[0]["text"] if isinstance(m.content, str) else str(m.content)
            else:
                c = self._to_anthropic_content(m.content)
                msgs.append({"role": m.role, "content": c})
        resp = client.messages.create(
            model=model,
            system=system,
            messages=msgs,
            **kwargs,
        )
        text = resp.content[0].text if resp.content else ""
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens} if resp.usage else None
        return ChatResponse(content=text, model=resp.model_id, usage=usage, raw=resp)

    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        return self.chat(messages, model=model or self.default_vision_model, **kwargs)

    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        raise NotImplementedError("Anthropic does not offer a public embedding API; use OpenAI or sentence-transformers")
