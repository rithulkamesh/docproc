"""RAG backend factory — creates RAG from config (single store + primary AI)."""

import logging
from typing import Optional

from docproc.config import get_config
from docproc.providers.factory import get_provider
from docproc.rag.base import RAGBackend
from docproc.stores.factory import create_store

logger = logging.getLogger(__name__)

_RAG: Optional[RAGBackend] = None


def create_rag() -> Optional[RAGBackend]:
    """Create RAG backend from config. Uses single database + primary AI provider."""
    global _RAG
    cfg = get_config()
    store = create_store(cfg.database)
    provider = get_provider(cfg.primary_ai)
    if store is None or provider is None:
        return None
    if cfg.rag.backend == "clara":
        from docproc.rag.clara_rag import ClaraRAG
        _RAG = ClaraRAG()
    else:
        from docproc.rag.embedding_rag import EmbeddingRAG
        _RAG = EmbeddingRAG(
            store=store,
            embed_provider=provider,
            llm_provider=provider,
            namespace=cfg.rag.namespace,
            chunk_size=cfg.rag.chunk_size,
            ingest_config=getattr(cfg, "ingest", None),
        )
    return _RAG


def get_rag() -> Optional[RAGBackend]:
    """Get RAG backend. Creates from config if not yet initialized."""
    global _RAG
    if _RAG is None:
        _RAG = create_rag()
    return _RAG
