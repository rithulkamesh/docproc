#!/usr/bin/env python3
"""Extract using in-memory OpenAI config (no YAML file).

Environment: OPENAI_API_KEY (and optional OPENAI base URL via docproc.yaml if you switch to from_config_path).

Usage:
  python minimal_in_memory.py input.pdf output.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from docproc import Docproc


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python minimal_in_memory.py input.pdf output.md", file=sys.stderr)
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    Docproc.with_openai().extract_to_file(inp, out)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
