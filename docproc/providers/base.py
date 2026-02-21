"""Base provider interface for LLM and embedding models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str | List[dict]  # str or list of content blocks (e.g. text + image)


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str
    model: str
    usage: Optional[dict] = None
    raw: Optional[Any] = None


class ModelProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Azure, Anthropic, Ollama)."""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request."""
        pass

    @abstractmethod
    def chat_with_vision(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send vision-capable chat request (text + images)."""
        pass

    @abstractmethod
    def embed(self, texts: List[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
