"""Configuration loader from YAML/JSON file and environment."""

import os
from pathlib import Path
from typing import Optional

from docproc.config.schema import (
    AIProviderConfig,
    DatabaseConfig,
    DocProcConfig,
    IngestConfig,
    RAGConfig,
)

_CONFIG: Optional[DocProcConfig] = None


def load_config(path: Optional[str] = None) -> DocProcConfig:
    """Load configuration from file. Environment variables override file values.

    Searches (in order): path, DOCPROC_CONFIG, ./docproc.yaml, ./docproc.yml, ~/.docproc.yaml

    Args:
        path: Explicit config file path

    Returns:
        DocProcConfig instance
    """
    global _CONFIG
    candidates = []
    if path and os.path.exists(path):
        candidates = [path]
    else:
        candidates = [
            os.getenv("DOCPROC_CONFIG"),
            "docproc.yaml",
            "docproc.yml",
            os.path.expanduser("~/.docproc.yaml"),
        ]
    config_path = None
    raw: dict = {}
    for p in candidates:
        if p and os.path.exists(p):
            config_path = p
            with open(p) as f:
                import yaml
                raw = yaml.safe_load(f) or {}
            break
    # Default connection strings from env
    db = raw.get("database", {})

    database = DatabaseConfig(
        provider=os.getenv("DOCPROC_DATABASE_PROVIDER", db.get("provider", "memory")),
        connection_string=os.getenv("DATABASE_URL", db.get("connection_string")),
        collection_name=db.get("collection_name", "docproc"),
        table_name=db.get("table_name", "docproc_chunks"),
        embed_dim=int(db.get("embed_dim", 1536)),
        path=os.getenv("DOCPROC_DB_PATH", db.get("path")),
    )

    providers_raw = raw.get("ai_providers", [{"provider": "openai"}])
    if isinstance(providers_raw, list):
        ai_providers = [
            AIProviderConfig(
                provider=p.get("provider", "openai"),
                api_key=p.get("api_key"),
                base_url=p.get("base_url"),
                default_model=p.get("default_model"),
                default_vision_model=p.get("default_vision_model"),
                extra=p.get("extra", {}),
            )
            for p in providers_raw
        ]
    else:
        ai_providers = [AIProviderConfig(provider="openai")]

    rag_raw = raw.get("rag", {})
    rag = RAGConfig(
        backend=rag_raw.get("backend", "embedding"),
        top_k=int(rag_raw.get("top_k", 5)),
        chunk_size=int(rag_raw.get("chunk_size", 512)),
        namespace=rag_raw.get("namespace", "default"),
    )

    ingest_raw = raw.get("ingest", {})
    ingest = IngestConfig(
        sanitize=ingest_raw.get("sanitize", True),
        drop_exact_duplicates=ingest_raw.get("drop_exact_duplicates", True),
        drop_boilerplate=ingest_raw.get("drop_boilerplate", True),
        boilerplate_kinds=ingest_raw.get("boilerplate_kinds"),
    )

    _CONFIG = DocProcConfig(
        database=database,
        ai_providers=ai_providers,
        primary_ai=raw.get("primary_ai", ai_providers[0].provider if ai_providers else "openai"),
        rag=rag,
        ingest=ingest,
        dla=raw.get("dla", {"use_fallback": True}),
        config_path=config_path,
    )
    return _CONFIG


def get_config() -> DocProcConfig:
    """Get current config. Loads defaults if not yet loaded."""
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG
