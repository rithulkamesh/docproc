#!/usr/bin/env python3
"""Print extracted markdown to stdout (for RAG/LLM pipelines).

Uses default config discovery (DOCPROC_CONFIG, ./docproc.yaml, ~/.config/docproc/docproc.yml).
Set provider API keys per docs/CONFIGURATION.md.

Usage:
  python extract_string.py path/to/document.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from docproc import Docproc


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_string.py path/to/document.pdf", file=sys.stderr)
        sys.exit(1)
    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    md = Docproc.from_env().extract(inp)
    print(md, end="")


if __name__ == "__main__":
    main()
