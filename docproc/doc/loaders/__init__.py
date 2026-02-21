"""Multi-format document loaders."""

from docproc.doc.loaders.base import DocumentLoader, LoadedPage
from docproc.doc.loaders.factory import load_document, get_supported_extensions

__all__ = [
    "DocumentLoader",
    "LoadedPage",
    "load_document",
    "get_supported_extensions",
]
