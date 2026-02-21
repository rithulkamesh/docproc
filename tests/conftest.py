"""Pytest fixtures for docproc tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_config(tmp_path):
    """Write a minimal docproc YAML config to a temp file."""
    config_path = tmp_path / "docproc.yaml"
    config_path.write_text(
        """
primary_ai: ollama
ai_providers:
  - provider: ollama
    base_url: http://localhost:11434
    default_model: llava
    default_vision_model: llava
ingest:
  use_vision: false
  use_llm_refine: false
""",
        encoding="utf-8",
    )
    return str(config_path)


@pytest.fixture
def sample_docx(tmp_path):
    """Create a minimal valid DOCX file with 'Hello world' content."""
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello world")
    path = tmp_path / "sample.docx"
    doc.save(path)
    return path
