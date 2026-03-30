#!/usr/bin/env python3
"""Extract a document to markdown using a YAML config file.

Environment: set provider keys as in docs/CONFIGURATION.md (e.g. OPENAI_API_KEY for OpenAI).
You can start from docproc.example.yaml in the repo root.

Usage:
  python minimal_from_config.py docproc.yaml input.pdf output.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from docproc import Docproc


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python minimal_from_config.py [docproc.yaml] input.pdf output.md",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    inp = Path(sys.argv[2])
    out = Path(sys.argv[3])
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    Docproc.from_config_path(cfg_path).extract_to_file(inp, out)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
