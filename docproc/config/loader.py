"""Configuration loader from YAML/JSON file and environment."""

import os
from pathlib import Path
from typing import Optional

from docproc.config.schema import (
    AIConfig,
    AIProviderConfig,
    DatabaseConfig,
    docprocConfig,
    IngestConfig,
    RAGConfig,
)

_CONFIG: Optional[docprocConfig] = None

_VALID_DB_PROVIDERS = {"pgvector", "qdrant", "chroma", "faiss", "memory"}
_VALID_RAG_BACKENDS = {"clara", "embedding"}
_VALID_AI_PROVIDERS = {"openai", "azure", "anthropic", "ollama", "litellm"}


def _validate_config(cfg: docprocConfig) -> None:
    """Validate config schema; raises ValueError on invalid values."""
    if cfg.database.provider.lower() not in _VALID_DB_PROVIDERS:
        raise ValueError(
            f"database.provider must be one of {_VALID_DB_PROVIDERS}, got: {cfg.database.provider}"
        )
    if cfg.rag.backend.lower() not in _VALID_RAG_BACKENDS:
        raise ValueError(
            f"rag.backend must be one of {_VALID_RAG_BACKENDS}, got: {cfg.rag.backend}"
        )
    for i, p in enumerate(cfg.ai_providers):
        if p.provider.lower() not in _VALID_AI_PROVIDERS:
            raise ValueError(
                f"ai_providers[{i}].provider must be one of {_VALID_AI_PROVIDERS}, got: {p.provider}"
            )
    if cfg.primary_ai.lower() not in _VALID_AI_PROVIDERS:
        raise ValueError(
            f"primary_ai must be one of {_VALID_AI_PROVIDERS}, got: {cfg.primary_ai}"
        )
    if not any(p.provider.lower() == cfg.primary_ai.lower() for p in cfg.ai_providers):
        raise ValueError(
            f"primary_ai '{cfg.primary_ai}' not found in ai_providers"
        )


def load_config(path: Optional[str] = None) -> docprocConfig:
    """Load configuration from file. Environment variables override file values.

    Searches (in order): path, DOCPROC_CONFIG, ./docproc.yaml, ./docproc.yml, ~/.docproc.yaml

    Args:
        path: Explicit config file path

    Returns:
        docprocConfig instance
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
            os.path.expanduser("~/.config/docproc/docproc.yml"),
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

    ai_raw = raw.get("ai", {})
    providers_raw = raw.get("ai_providers", [{"provider": "openai"}])
    if isinstance(providers_raw, list):
        ai_providers = [
            AIProviderConfig(
                provider=p.get("provider", "openai"),
                api_key=p.get("api_key"),
                base_url=p.get("base_url"),
                default_model=p.get("default_model"),
                default_vision_model=p.get("default_vision_model"),
                timeout=int(p.get("timeout") or ai_raw.get("timeout") or os.getenv("AI_TIMEOUT", 60)),
                max_retries=int(p.get("max_retries") or ai_raw.get("max_retries") or os.getenv("AI_MAX_RETRIES", 3)),
                extra=p.get("extra", {}),
            )
            for p in providers_raw
        ]
    else:
        ai_providers = [AIProviderConfig(provider="openai")]

    # Env override for primary AI (e.g. DOCPROC_PRIMARY_AI=azure when .env has Azure keys)
    primary_ai_from_env = os.getenv("DOCPROC_PRIMARY_AI", "").strip().lower()
    if primary_ai_from_env:
        primary_ai = primary_ai_from_env
        # Ensure the chosen provider exists in the list so get_provider() can find it
        if not any(p.provider.lower() == primary_ai for p in ai_providers):
            ai_providers = list(ai_providers) + [AIProviderConfig(provider=primary_ai)]
    else:
        primary_ai = raw.get("primary_ai", ai_providers[0].provider if ai_providers else "openai")

    rag_raw = raw.get("rag", {})
    rag = RAGConfig(
        backend=rag_raw.get("backend", "clara"),
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
        use_vision=ingest_raw.get("use_vision", True),
    )

    _str_to_bool = lambda v: str(v).strip().lower() in ("1", "true", "yes")
    ai = AIConfig(
        provider=os.getenv("AI_PROVIDER") or ai_raw.get("provider"),
        model_primary=os.getenv("AI_MODEL_PRIMARY") or ai_raw.get("model_primary"),
        model_secondary=os.getenv("AI_MODEL_SECONDARY") or ai_raw.get("model_secondary"),
        temperature=float(os.getenv("AI_TEMPERATURE", ai_raw.get("temperature", 0.2))),
        timeout=int(os.getenv("AI_TIMEOUT", ai_raw.get("timeout", 60))),
        max_retries=int(os.getenv("AI_MAX_RETRIES", ai_raw.get("max_retries", 3))),
        disabled=_str_to_bool(os.getenv("AI_DISABLED", ai_raw.get("disabled", False))),
        eval_discrepancy_threshold=float(os.getenv("AI_EVAL_DISCREPANCY_THRESHOLD", ai_raw.get("eval_discrepancy_threshold", 10.0))),
        max_answer_tokens=int(os.getenv("AI_MAX_ANSWER_TOKENS", ai_raw.get("max_answer_tokens", 2000))),
    )

    _CONFIG = docprocConfig(
        database=database,
        ai_providers=ai_providers,
        primary_ai=primary_ai,
        rag=rag,
        ingest=ingest,
        ai=ai,
        dla=raw.get("dla", {"use_fallback": True}),
        config_path=config_path,
    )
    _validate_config(_CONFIG)
    return _CONFIG


def get_config() -> docprocConfig:
    """Get current config. Loads defaults if not yet loaded."""
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG
