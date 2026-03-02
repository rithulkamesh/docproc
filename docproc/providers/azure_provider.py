"""Azure OpenAI provider."""

import os
from typing import List, Optional

from docproc.providers.base import ChatMessage, ChatResponse, ModelProvider


class AzureOpenAIProvider(ModelProvider):
    """Azure OpenAI provider for chat and embeddings.

    Chat: AZURE_OPENAI_DEPLOYMENT (or config default_model).
    Embeddings: AZURE_OPENAI_EMBEDDING_DEPLOYMENT (or config extra.embedding_deployment).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        default_model: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.default_model = default_model or self.deployment
        self.embedding_deployment = (
            embedding_deployment
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        from openai import AzureOpenAI
        client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.endpoint, api_version="2024-02-15-preview")
        model = model or self.default_model
        msgs = [{"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)} for m in messages]
        resp = client.chat.completions.create(model=model, messages=msgs, **kwargs)
        content = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else None
        return ChatResponse(content=content, model=resp.model, usage=usage, raw=resp)

    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        from openai import AzureOpenAI
        client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.endpoint, api_version="2024-02-15-preview")
        model = model or self.default_model
        msgs = []
        for m in messages:
            if isinstance(m.content, list):
                msgs.append({"role": m.role, "content": m.content})
            else:
                msgs.append({"role": m.role, "content": str(m.content)})
        resp = client.chat.completions.create(model=model, messages=msgs, **kwargs)
        content = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else None
        return ChatResponse(content=content, model=resp.model, usage=usage, raw=resp)

    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        from openai import AzureOpenAI
        deployment = model or self.embedding_deployment
        client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.endpoint, api_version="2024-02-15-preview")
        resp = client.embeddings.create(input=texts, model=deployment, **kwargs)
        return [d.embedding for d in resp.data]
