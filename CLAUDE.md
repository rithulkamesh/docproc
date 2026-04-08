# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_facade.py

# Run a single test by name
pytest tests/test_config.py::test_parse_config_defaults -v

# Build the package
uv build

# CLI usage
docproc --file input.pdf -o output.md
docproc --file input.pdf -o output.md --config docproc.yaml
docproc init-config --env .env
docproc completions [bash|zsh]
```

## Architecture

docproc is a document extraction library + CLI. It reads PDF/DOCX/PPTX/XLSX files and outputs markdown via a three-stage pipeline:

1. **Load** — `docproc/doc/loaders/` contains format-specific loaders (PyMuPDF for PDF, python-docx, python-pptx, openpyxl). All loaders implement the same base interface in `loaders/base.py` and are selected by `loaders/factory.py`.

2. **Extract/Vision** — `docproc/extractors/vision_llm.py` sends PDF page images to a vision-capable LLM (enabled when `ingest.use_vision: true`). Falls back to native text on any connection/provider error.

3. **Refine** — `docproc/refiners/llm_refine.py` passes extracted text through an LLM to clean markdown and format LaTeX (enabled when `ingest.use_llm_refine: true`).

**Entry points:**
- `docproc/pipeline.py` — `extract_document_to_text()` is the core function used by both the CLI and library. It orchestrates vision extraction → text fallback → LLM refine.
- `docproc/facade.py` — `Docproc` class wraps the pipeline with instance-scoped config. Factory classmethods: `with_openai()`, `from_config_path()`, `from_env()`.
- `docproc/bin/cli.py` — CLI entry point registered as `docproc` in `pyproject.toml`.

**Config system** (`docproc/config/`):
- `schema.py` defines `docprocConfig` (dataclass) with sub-configs: `DatabaseConfig`, `AIProviderConfig`, `IngestConfig`, `RAGConfig`, `AIConfig`.
- `loader.py` has two functions: `parse_config()` (pure, returns a new config, does NOT update global state) and `load_config()` (sets the process-wide singleton used by `get_config()`). Use `parse_config()` in tests and library code; `load_config()` in CLI.
- Config is resolved from (in order): explicit path → `DOCPROC_CONFIG` env → `./docproc.yaml` → `./docproc.yml` → `~/.config/docproc/docproc.yml`.

**Provider system** (`docproc/providers/`):
- `factory.py` — `get_provider()` creates and caches provider instances. Bypasses cache when a `config` argument is passed (use this in tests).
- Supported providers: `openai`, `azure`, `anthropic`, `ollama`, `litellm`.
- All providers implement `ModelProvider` base class in `base.py`.

**Demo** (`demo/`) is a separate full-stack application (Go API + React UI + PostgreSQL/PgVector + RabbitMQ) that invokes the `docproc` CLI as a subprocess when documents are uploaded. It is not part of the Python library.

## Config search and env overrides

Key env vars: `DOCPROC_CONFIG`, `DOCPROC_PRIMARY_AI`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, `DATABASE_URL`, `AI_DISABLED`.

For local dev without a config file, copy `.env.example` → `.env` and run `docproc init-config` to generate `~/.config/docproc/docproc.yml`.
