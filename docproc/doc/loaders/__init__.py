"""Multi-format document loaders."""

from docproc.doc.loaders.base import DocumentLoader, LoadedPage
from docproc.doc.loaders.factory import (
    get_full_text,
    get_loader,
    get_page_count,
    get_supported_extensions,
    load_document,
)

__all__ = [
    "DocumentLoader",
    "LoadedPage",
    "get_full_text",
    "get_loader",
    "get_page_count",
    "get_supported_extensions",
    "load_document",
]
