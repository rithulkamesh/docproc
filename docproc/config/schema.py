"""Configuration schema and defaults.

Single database provider. Multiple AI providers. Config-driven architecture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatabaseConfig:
    """Single database provider configuration.

    Only one database is used at a time (single source of truth).
    """

    provider: str  # pgvector | qdrant | chroma | faiss | memory
    connection_string: Optional[str] = None
    collection_name: str = "docproc"
    table_name: str = "docproc_chunks"
    embed_dim: int = 1536
    path: Optional[str] = None  # for chroma/faiss persistent storage


@dataclass
class AIProviderConfig:
    """Configuration for a single AI provider."""

    provider: str  # openai | azure | anthropic | ollama | litellm
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    default_vision_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestConfig:
    """Document ingestion: sanitization and deduplication."""

    sanitize: bool = True
    drop_exact_duplicates: bool = True
    drop_boilerplate: bool = True
    boilerplate_kinds: Optional[List[str]] = None  # e.g. ["thank_you", "questions", "blank"]


@dataclass
class RAGConfig:
    """RAG backend configuration."""

    backend: str  # embedding | clara
    top_k: int = 5
    chunk_size: int = 512
    namespace: str = "default"


@dataclass
class DocProcConfig:
    """DocProc configuration — single source of truth.

    Attributes:
        database: Single database provider config
        ai_providers: List of AI providers (multiple allowed)
        primary_ai: Provider ID to use for chat/embed by default
        rag: RAG backend config
        dla: Document layout analysis config
    """

    database: DatabaseConfig
    ai_providers: List[AIProviderConfig]
    primary_ai: str = "openai"
    rag: RAGConfig = field(default_factory=lambda: RAGConfig(backend="embedding"))
    ingest: IngestConfig = field(default_factory=IngestConfig)
    dla: Dict[str, Any] = field(default_factory=lambda: {"use_fallback": True})
    config_path: Optional[str] = None
