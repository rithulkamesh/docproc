"""Tests for parse_config isolation and Docproc facade."""

import docproc.config.loader as loader_module

from docproc import Docproc
from docproc.config.loader import parse_config


def test_parse_config_does_not_set_global(tmp_config):
    """parse_config returns config without updating the module singleton."""
    loader_module._CONFIG = None
    cfg = parse_config(tmp_config)
    assert cfg.primary_ai == "ollama"
    assert loader_module._CONFIG is None


def test_docproc_from_config_path_extracts_docx(tmp_config, sample_docx):
    """Docproc uses instance config; DOCX extracts without global load_config."""
    loader_module._CONFIG = None
    dp = Docproc.from_config_path(tmp_config)
    text = dp.extract(sample_docx)
    assert "Hello world" in text


def test_docproc_extract_to_file(tmp_config, sample_docx, tmp_path):
    """extract_to_file writes markdown; page comment for DOCX may be 0 or 1."""
    dp = Docproc.from_config_path(tmp_config)
    out = tmp_path / "out.md"
    path = dp.extract_to_file(sample_docx, out, include_page_comment=True)
    assert path == out.resolve()
    body = out.read_text(encoding="utf-8")
    assert "Hello world" in body


def test_docproc_explicit_config(sample_docx, tmp_config):
    """Docproc(config=parse_config(...)) works without load_config."""
    cfg = parse_config(tmp_config)
    dp = Docproc(config=cfg)
    assert "Hello world" in dp.extract(sample_docx)


def test_load_config_still_sets_global(tmp_config):
    """load_config continues to populate get_config()."""
    from docproc.config.loader import get_config, load_config

    loader_module._CONFIG = None
    load_config(tmp_config)
    assert get_config().primary_ai == "ollama"
