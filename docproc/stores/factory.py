"""Vector store factory — creates a single store from config.

Single source of truth: only one database provider is used.
"""

import logging
from typing import Optional

from docproc.config.schema import DatabaseConfig
from docproc.stores.base import VectorStore

logger = logging.getLogger(__name__)


def create_store(config: Optional[DatabaseConfig] = None) -> Optional[VectorStore]:
    """Create the configured vector store (single instance).

    Args:
        config: Database config; uses get_config() if None

    Returns:
        VectorStore instance or None if creation fails
    """
    if config is None:
        from docproc.config import get_config
        config = get_config().database

    provider = config.provider.lower()
    try:
        if provider == "pgvector":
            return _create_pgvector(config)
        if provider == "qdrant":
            return _create_qdrant(config)
        if provider == "chroma":
            return _create_chroma(config)
        if provider == "faiss":
            return _create_faiss(config)
        if provider == "memory":
            return _create_memory(config)
        logger.warning(f"Unknown database provider: {provider}")
        return None
    except Exception as e:
        logger.error(f"Failed to create store {provider}: {e}")
        return None


def _create_pgvector(config: DatabaseConfig) -> VectorStore:
    """Create PgVector store."""
    import os
    from docproc.stores.pgvector_store import PgVectorStore
    conn = config.connection_string or os.getenv("DATABASE_URL", "postgresql://localhost/docproc")
    return PgVectorStore(connection_string=conn, table_name=config.table_name)


def _create_qdrant(config: DatabaseConfig) -> VectorStore:
    """Create Qdrant store."""
    import os
    from docproc.stores.qdrant_store import QdrantStore
    url = config.connection_string or os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantStore(url=url, collection_name=config.collection_name, embed_dim=config.embed_dim)


def _create_chroma(config: DatabaseConfig) -> VectorStore:
    """Create Chroma store."""
    from docproc.stores.chroma_store import ChromaStore
    path = config.path or config.connection_string
    return ChromaStore(
        collection_name=config.collection_name,
        persist_directory=path,
        embed_dim=config.embed_dim,
    )


def _create_faiss(config: DatabaseConfig) -> VectorStore:
    """Create FAISS store."""
    from docproc.stores.faiss_store import FAISSStore
    path = config.path or config.connection_string
    return FAISSStore(
        index_path=path,
        embed_dim=config.embed_dim,
    )


def _create_memory(config: DatabaseConfig) -> VectorStore:
    """Create in-memory store (ephemeral)."""
    from docproc.stores.memory_store import MemoryStore
    return MemoryStore(embed_dim=config.embed_dim)
