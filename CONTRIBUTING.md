# Contributing to docproc

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Python 3.11+

## Setup

```bash
git clone https://github.com/rithulkamesh/docproc.git
cd docproc
uv sync --extra dev
```

## Running tests

```bash
uv run pytest tests -v
```

## Code style

No strict linter enforced. Consider using [black](https://black.readthedocs.io/) or [ruff](https://docs.astral.sh/ruff/) for formatting.

## Pull requests

1. Fork the repo and create a branch
2. Make your changes
3. Run tests: `uv run pytest tests -v`
4. Open a PR with a clear description
5. Ensure CI passes
