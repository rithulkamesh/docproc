# DocProc Architecture

## Overview

DocProc extracts content from documents (PDF, DOCX, PPTX, XLSX), optionally refines it with LLMs, and indexes it for RAG queries.

## Pipeline flow

```
Document (PDF/DOCX/PPTX/XLSX)
    -> Load (get_full_text or vision extract for PDF images)
    -> Optional LLM refine (markdown, LaTeX)
    -> Sanitize & dedupe
    -> Output (.md for CLI) or Index (RAG for API)
```

## Modules

| Module | Purpose |
|--------|---------|
| `docproc/doc/loaders` | Load documents, extract full text. PDF uses PyMuPDF; DOCX/PPTX/XLSX use python-docx, python-pptx, openpyxl. |
| `docproc/extractors` | Vision LLM extraction for PDF embedded images (Azure Vision or vision-capable LLM). |
| `docproc/refiners` | LLM refinement: clean markdown, LaTeX math, remove boilerplate. |
| `docproc/providers` | AI providers: OpenAI, Azure, Anthropic, Ollama, LiteLLM. |
| `docproc/sanitize` | Text sanitization and deduplication. |
| `docproc/pipeline` | Shared extraction pipeline (extract_document_to_text) used by CLI and API. |
| `docproc/api` | FastAPI server: upload, documents, query, models. |
| `docproc/rag` | RAG backends: embedding-based or CLaRa. |
| `docproc/stores` | Vector stores: PgVector, Qdrant, Chroma, FAISS, memory. |

## Configuration

- **docproc.yaml**: Single config file. One database, multiple AI providers, one primary AI.
- **Environment overrides**: `DOCPROC_CONFIG`, `DATABASE_URL`, `OPENAI_API_KEY`, `AZURE_OPENAI_*`, etc.
- See [CONFIGURATION.md](CONFIGURATION.md) for the full schema.

## CLI vs API

- **CLI** (`docproc --file input.pdf -o output.md`): Runs the pipeline locally, writes to .md. No server, no RAG.
- **API** (`docproc-serve`): Accepts uploads, runs the pipeline in background, indexes to vector store, serves query endpoint.
