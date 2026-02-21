"""Data sanitization and deduplication for document content."""

from docproc.sanitize.sanitizer import sanitize_text, SanitizerConfig
from docproc.sanitize.dedupe import deduplicate_chunks, deduplicate_texts, BoilerplateKind, is_boilerplate

__all__ = [
    "sanitize_text",
    "SanitizerConfig",
    "deduplicate_chunks",
    "deduplicate_texts",
    "BoilerplateKind",
    "is_boilerplate",
]
