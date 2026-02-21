"""Unit tests for config loader and schema."""

import pytest

from docproc.config.loader import load_config


def test_load_config_with_explicit_path(tmp_config):
    """load_config with explicit path loads the file."""
    cfg = load_config(tmp_config)
    assert cfg.primary_ai == "ollama"
    assert len(cfg.ai_providers) == 1
    assert cfg.ai_providers[0].provider == "ollama"
    assert cfg.config_path == tmp_config


def test_load_config_minimal(tmp_config):
    """load_config with minimal file uses schema defaults for missing keys."""
    cfg = load_config(tmp_config)
    assert cfg.rag.backend == "clara"
    assert cfg.ingest.use_vision is False  # from our fixture


def test_load_config_rag_schema_defaults(tmp_config):
    """load_config applies schema defaults for rag when not in file."""
    cfg = load_config(tmp_config)
    assert cfg.rag.backend in ("clara", "embedding")
    assert cfg.rag.top_k == 5
    assert cfg.rag.chunk_size == 512
