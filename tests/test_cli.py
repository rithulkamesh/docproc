"""Smoke tests for CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help():
    """docproc --help exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "docproc.bin.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "output" in result.stdout.lower() or "file" in result.stdout.lower()


def test_cli_nonexistent_file():
    """docproc with nonexistent file exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "docproc.bin.cli", "--file", "/nonexistent/file.pdf", "-o", "out.md"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_extract_docx_to_md(sample_docx, tmp_config, tmp_path):
    """docproc extracts DOCX to markdown with config."""
    out_md = tmp_path / "output.md"
    result = subprocess.run(
        [
            sys.executable, "-m", "docproc.bin.cli",
            "--file", str(sample_docx),
            "-o", str(out_md),
            "--config", tmp_config,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Hello" in content or "hello" in content
