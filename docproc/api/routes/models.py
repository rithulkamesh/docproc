"""Models and provider listing. Reflects config (docproc.yaml)."""

from fastapi import APIRouter

from docproc.config import get_config

router = APIRouter()


@router.get("")
async def list_models():
    """List configured AI providers and supported database options."""
    cfg = get_config()
    return {
        "primary_ai": cfg.primary_ai,
        "database": cfg.database.provider,
        "ai_providers": [p.provider for p in cfg.ai_providers],
        "database_options": ["pgvector", "qdrant", "chroma", "faiss", "memory"],
        "ai_options": ["openai", "azure", "anthropic", "ollama", "litellm"],
    }
