"""Unit tests for document loaders."""

import pytest

from docproc.doc.loaders import (
    get_supported_extensions,
    get_loader,
    get_full_text,
    load_document,
)


def test_get_supported_extensions():
    """get_supported_extensions returns expected list."""
    exts = get_supported_extensions()
    assert ".docx" in exts
    assert ".pdf" in exts
    assert ".pptx" in exts
    assert ".xlsx" in exts
    assert len(exts) >= 4


def test_get_loader_raises_unsupported():
    """get_loader raises for unsupported format."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name
    try:
        from pathlib import Path
        with pytest.raises(ValueError, match="Unsupported format"):
            get_loader(Path(path))
    finally:
        import os
        os.unlink(path)


def test_get_full_text_docx(sample_docx):
    """get_full_text extracts content from DOCX fixture."""
    text = get_full_text(sample_docx)
    assert "Hello world" in text or "Hello" in text


def test_load_document_docx(sample_docx):
    """load_document yields pages from DOCX."""
    pages = list(load_document(sample_docx))
    assert len(pages) >= 1
    assert pages[0].text or any(r.content for r in pages[0].regions)
