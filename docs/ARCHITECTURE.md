# docproc Architecture

## Overview

docproc is a **CLI-only** document processor: it reads a file (PDF, DOCX, PPTX, XLSX), extracts content (native text + optional vision for embedded images), optionally refines it with an LLM, and writes markdown to a file. It does not run a server, store documents, or perform RAG.

The **demo** (in `demo/`) is a separate full-stack app: Go API, React UI, PostgreSQL + PgVector, LocalStack, RabbitMQ. It invokes the docproc CLI when a document is uploaded. See [demo/README.md](../demo/README.md).

## Pipeline flow

```
Document (PDF/DOCX/PPTX/XLSX)
    -> Load (get_full_text or vision extract for PDF images)
    -> Optional LLM refine (markdown, LaTeX)
    -> Sanitize & dedupe
    -> Output: .md file (CLI)
```

## Modules (docproc)

| Module | Purpose |
|--------|---------|
| `docproc/doc/loaders` | Load documents, extract full text. PDF uses PyMuPDF; DOCX/PPTX/XLSX use python-docx, python-pptx, openpyxl. |
| `docproc/extractors` | Vision LLM extraction for PDF embedded images (Azure Vision or vision-capable LLM). |
| `docproc/refiners` | LLM refinement: clean markdown, LaTeX math, remove boilerplate. |
| `docproc/providers` | AI providers: OpenAI, Azure, Anthropic, Ollama, LiteLLM. |
| `docproc/sanitize` | Text sanitization and deduplication. |
| `docproc/pipeline` | Extraction pipeline (`extract_document_to_text`) used by the CLI. |
| `docproc/config` | Config loader and schema (`docproc.yaml`). |

## Configuration

- **docproc.yaml**: Single config file. AI providers and ingest options (vision, LLM refine). No database required for the CLI.
- **Environment:** `DOCPROC_CONFIG`, `OPENAI_API_KEY`, `AZURE_OPENAI_*`, etc. See [CONFIGURATION.md](CONFIGURATION.md).

## CLI

- **Extract:** `docproc --file input.pdf -o output.md` — Runs the pipeline and writes markdown. No server, no RAG.
- **init-config:** `docproc init-config [--env .env]` — Writes `~/.config/docproc/docproc.yml` from environment.
- **completions:** `docproc completions [bash|zsh]` — Shell completion script.
