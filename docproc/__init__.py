"""docproc — document to markdown for AI pipelines (CLI and library)."""

from docproc.config import docprocConfig, get_config, load_config, parse_config
from docproc.facade import Docproc
from docproc.pipeline import extract_document_to_text

__all__ = [
    "Docproc",
    "docprocConfig",
    "extract_document_to_text",
    "get_config",
    "load_config",
    "parse_config",
]
