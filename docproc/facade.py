"""High-level API: construct with config, extract to string or file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from docproc.config.loader import parse_config
from docproc.config.schema import (
    AIProviderConfig,
    DatabaseConfig,
    IngestConfig,
    RAGConfig,
    docprocConfig,
)
from docproc.doc.loaders import get_page_count
from docproc.pipeline import extract_document_to_text


class Docproc:
    """Document processor with instance-scoped config (no reliance on get_config() global)."""

    def __init__(self, config: Optional[docprocConfig] = None) -> None:
        self._config = config if config is not None else parse_config()

    @classmethod
    def from_config_path(cls, path: str | Path) -> Docproc:
        """Load YAML/JSON from path via the same rules as parse_config (file + env overrides)."""
        return cls(config=parse_config(path))

    @classmethod
    def from_env(cls) -> Docproc:
        """Load config from DOCPROC_CONFIG or default search paths (same as parse_config(None))."""
        return cls(config=parse_config(None))

    @classmethod
    def with_openai(
        cls,
        *,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_vision_model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Docproc:
        """Minimal in-memory config for OpenAI (vision + refine use primary_ai openai)."""
        key = api_key or os.getenv("OPENAI_API_KEY")
        cfg = docprocConfig(
            database=DatabaseConfig(provider="memory"),
            ai_providers=[
                AIProviderConfig(
                    provider="openai",
                    api_key=key,
                    base_url=base_url,
                    default_model=default_model,
                    default_vision_model=default_vision_model or default_model,
                )
            ],
            primary_ai="openai",
            rag=RAGConfig(backend="embedding"),
            ingest=IngestConfig(),
        )
        return cls(config=cfg)

    def extract(
        self,
        path: str | Path,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Extract document to markdown text (same pipeline as the CLI)."""
        return extract_document_to_text(
            Path(path),
            config=self._config,
            progress_callback=progress_callback,
        )

    def extract_to_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        include_page_comment: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Write extracted markdown to a file; optionally prefix with <!-- PAGES: n --> like the CLI."""
        inp = Path(input_path)
        out = Path(output_path)
        full_text = self.extract(inp, progress_callback=progress_callback)
        if include_page_comment:
            try:
                num_pages = get_page_count(inp)
            except Exception:
                num_pages = 0
            if num_pages > 0:
                full_text = f"<!-- PAGES: {num_pages} -->\n" + full_text
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full_text, encoding="utf-8")
        return out.resolve()
