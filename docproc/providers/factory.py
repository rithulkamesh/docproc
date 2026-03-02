"""AI provider factory — creates provider instances from config."""

import logging
import os
from typing import Dict, Optional

from docproc.config.schema import AIProviderConfig, docprocConfig
from docproc.providers.base import ModelProvider

logger = logging.getLogger(__name__)

_PROVIDERS: Dict[str, ModelProvider] = {}


def create_provider(config: AIProviderConfig) -> Optional[ModelProvider]:
    """Create a single AI provider from config.

    Args:
        config: AIProviderConfig instance

    Returns:
        ModelProvider or None
    """
    provider = config.provider.lower()
    try:
        if provider == "openai":
            return _create_openai(config)
        if provider == "azure":
            return _create_azure(config)
        if provider == "anthropic":
            return _create_anthropic(config)
        if provider == "ollama":
            return _create_ollama(config)
        if provider == "litellm":
            return _create_litellm(config)
        logger.warning(f"Unknown AI provider: {provider}")
        return None
    except Exception as e:
        logger.error(f"Failed to create provider {provider}: {e}")
        return None


def get_primary_provider() -> Optional[ModelProvider]:
    """Get the primary AI provider for grading/generation (respects AI disabled and model_primary)."""
    from docproc.config import get_config
    cfg = get_config()
    if cfg.ai.disabled:
        return None
    provider_id = cfg.ai.model_primary or cfg.primary_ai
    return get_provider(provider_id) or get_provider()


def get_provider(provider_id: Optional[str] = None) -> Optional[ModelProvider]:
    """Get provider by ID. Uses primary_ai from config if provider_id is None.

    Caches provider instances.
    """
    global _PROVIDERS
    if provider_id is None:
        from docproc.config import get_config
        cfg = get_config()
        provider_id = cfg.primary_ai
    if provider_id in _PROVIDERS:
        return _PROVIDERS[provider_id]
    from docproc.config import get_config
    cfg = get_config()
    for pc in cfg.ai_providers:
        if pc.provider.lower() == provider_id.lower():
            prov = create_provider(pc)
            if prov:
                _PROVIDERS[provider_id] = prov
            return prov
    return None


def _create_openai(config: AIProviderConfig) -> ModelProvider:
    from docproc.providers.openai_provider import OpenAIProvider
    return OpenAIProvider(
        api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=config.base_url or os.getenv("OPENAI_BASE_URL"),
        default_model=os.getenv("OPENAI_DEFAULT_MODEL") or config.default_model or "gpt-4o",
        default_vision_model=os.getenv("OPENAI_DEFAULT_VISION_MODEL") or config.default_vision_model or "gpt-4o",
    )


def _create_azure(config: AIProviderConfig) -> ModelProvider:
    from docproc.providers.azure_provider import AzureOpenAIProvider
    extra = config.extra or {}
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or config.default_model
    return AzureOpenAIProvider(
        api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        endpoint=config.base_url or os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment=deployment,
        embedding_deployment=extra.get("embedding_deployment") or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )


def _create_anthropic(config: AIProviderConfig) -> ModelProvider:
    from docproc.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider(
        api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
        default_model=config.default_model or "claude-sonnet-4-20250514",
        default_vision_model=config.default_vision_model or "claude-sonnet-4-20250514",
    )


def _create_ollama(config: AIProviderConfig) -> ModelProvider:
    from docproc.providers.ollama_provider import OllamaProvider
    return OllamaProvider(
        base_url=config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        default_model=config.default_model or "llava",
        default_vision_model=config.default_vision_model or "llava",
    )


def _create_litellm(config: AIProviderConfig) -> ModelProvider:
    from docproc.providers.litellm_provider import LiteLLMProvider
    return LiteLLMProvider(
        default_model=config.default_model or "gpt-4o",
        default_vision_model=config.default_vision_model or "gpt-4o",
    )
