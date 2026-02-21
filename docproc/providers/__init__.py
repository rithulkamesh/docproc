"""Multi-provider LLM support. Use factory.get_provider() for config-driven access."""

from docproc.providers.base import ModelProvider, ChatMessage, ChatResponse
from docproc.providers.factory import create_provider, get_provider
from docproc.providers.openai_provider import OpenAIProvider
from docproc.providers.anthropic_provider import AnthropicProvider
from docproc.providers.ollama_provider import OllamaProvider
from docproc.providers.azure_provider import AzureOpenAIProvider
from docproc.providers.litellm_provider import LiteLLMProvider

__all__ = [
    "ModelProvider",
    "ChatMessage",
    "ChatResponse",
    "create_provider",
    "get_provider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LiteLLMProvider",
]
