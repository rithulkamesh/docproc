# DocProc

**Document Intelligence Platform** — Extract, refine, and query documents with vision LLMs, config-driven RAG, and a NotebookLM-style UI.

## Motivation

I learn by asking questions. Not surface-level ones. The deep, obsessive "why"s that most materials never bother to answer. When my peers were studying from slides and PDFs, I sat there stuck. I couldn't absorb content I wasn't allowed to interrogate. Documents don't talk back. They don't explain the intuition, the connections, the *why*. Tools like NotebookLM couldn't help either: they don't understand images inside the data source, so those parts show up blank. Most of my slides were visual or text as screenshots. I was left with nothing.

So I built something for myself. A library that extracts content from any document — slides, papers, textbooks — and lets me use AI to actually ask. *Why does this work? What's the reasoning here? How does this connect to that thing from last week?* For the first time, static documents became something I could learn from. Not by re-reading. By *conversing*.

I'm open-sourcing it because I'm probably not the only one who learns this way.

---

## Features

- **Full content extraction** — Native PDF/DOCX/PPTX/XLSX text plus **vision** for every embedded image (equations, diagrams, labels).
- **Azure AI Vision** — Computer Vision Describe + Read (OCR) for images when Azure OpenAI vision isn’t available.
- **LLM refinement** — Optional pass to clean extracted text: markdown, LaTeX math, boilerplate removed, before indexing.
- **Config-driven** — Single `docproc.yaml`: one vector store, multiple AI providers.
- **Stores** — PgVector, Qdrant, Chroma, FAISS, or in-memory.
- **Providers** — OpenAI, Azure, Anthropic, Ollama, LiteLLM.
- **RAG** — Embedding-based or Apple CLaRa.
- **API + UI** — FastAPI, Streamlit frontend (per-file progress, Library + Chat), Open WebUI–compatible routes.
- **Async upload** — Background processing with per-file progress bar; parallel image extraction.

## Architecture

```
Upload (PDF/DOCX/PPTX/XLSX)
    → Extract (native text + vision for images)
    → Refine (LLM: markdown, LaTeX, no boilerplate) [optional]
    → Sanitize & dedupe
    → Index into vector store
    → Query via RAG
```

- **Config:** `docproc.yaml` selects one database and one primary AI provider.
- **Vision:** PDFs use native text layer; embedded images go to Azure Vision (Describe + Read) or a vision LLM.
- **Refinement:** With `ingest.use_llm_refine: true`, extracted text is cleaned and formatted before storage.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full schema.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/rithulkamesh/docproc.git && cd docproc
uv sync --python 3.12

# 2. Config and env
cp docproc.example.yaml docproc.yaml
cp .env.example .env
# Edit docproc.yaml (database + primary_ai) and .env (API keys, DATABASE_URL)

# 3. Start vector DB (e.g. Qdrant)
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Run API
docproc-serve

# 5. Run frontend (another terminal)
DOCPROC_API_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

Open http://localhost:8501 — upload a PDF, watch per-file progress, then chat or browse the Library.

## Configuration

Create `docproc.yaml` in the project root (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)):

```yaml
database:
  provider: pgvector   # pgvector | qdrant | chroma | faiss | memory
  # connection_string from DATABASE_URL or set here

ai_providers:
  - provider: azure    # or openai, anthropic, ollama, litellm
primary_ai: azure

rag:
  backend: embedding
  top_k: 5
  chunk_size: 512

ingest:
  use_vision: true      # PDF: extract text + vision for images
  use_llm_refine: true   # Clean markdown, LaTeX, remove boilerplate
```

Secrets (API keys, endpoints) come from environment variables or `.env`. See [.env.example](.env.example).

## Installation

### CLI

```bash
# With uv (recommended — isolated install, adds docproc to PATH)
uv tool install git+https://github.com/rithulkamesh/docproc.git

# Or with pip
pip install git+https://github.com/rithulkamesh/docproc.git
```

Then `docproc --file input.pdf -o output.md`. CLI uses the same config and providers as the server (OpenAI, Azure, Anthropic, Ollama, LiteLLM). For Ollama: `ollama pull llava && ollama serve` and use `docproc.cli.yaml` or `primary_ai: ollama`.

### Server (API + RAG + frontend)

```bash
uv tool install 'docproc[server] @ git+https://github.com/rithulkamesh/docproc.git'
# or pip install docproc[server]
```

### From source (dev)

```bash
git clone https://github.com/rithulkamesh/docproc.git && cd docproc
uv sync --python 3.12
# Run: uv run docproc --file input.pdf -o output.md
# Or install: uv pip install -e .
```

## Usage

### API

```bash
DOCPROC_CONFIG=docproc.yaml docproc-serve
# API at http://localhost:8000
```

Endpoints: `POST /documents/upload`, `GET /documents/`, `GET /documents/{id}`, `POST /query`, `GET /models`. Upload returns immediately with a document ID; processing runs in the background. Poll `GET /documents/{id}` for `status` and `progress` (page/total/message).

### Frontend

```bash
DOCPROC_API_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

- **Sources** — Refresh, list documents, upload (PDF/DOCX/PPTX/XLSX). Progress bar updates while the file is processed.
- **Library** — Select a document to view full extracted/refined text.
- **Chat** — Ask questions; answers are grounded in your documents.

### Docker

**From GHCR (recommended):**

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx ghcr.io/rithulkamesh/docproc:latest
```

**Build locally (standalone, in-memory DB):**

```bash
docker build -t docproc:2.0 .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx docproc:2.0
```

**Full stack (API + frontend + Postgres + Qdrant):**

```bash
cp docproc.example.yaml docproc.yaml
# Set database.provider: pgvector or qdrant and configure .env
docker-compose up
# API: 8000, Frontend: 8501, Postgres: 5432, Qdrant: 6333
```

### Open WebUI

Point Open WebUI to `http://localhost:8000/api` for OpenAI-compatible chat backed by your documents.

### CLI

```bash
# Requires Ollama + vision model (ollama pull llava)
cp docproc.cli.yaml docproc.yaml
docproc --file input.pdf -o output.md
```

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Config schema, database options, AI providers, ingest, RAG |
| [docs/AZURE_SETUP.md](docs/AZURE_SETUP.md) | Azure OpenAI + Azure AI Vision (Describe + Read), credentials |

## Environment

- `DOCPROC_CONFIG` — Path to config file (default: `docproc.yaml`).
- `DOCPROC_API_URL` — API base URL for the Streamlit frontend (default: `http://localhost:8000`).
- `DATABASE_URL` — Overrides `database.connection_string` (e.g. Postgres).
- Provider-specific: `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_VISION_ENDPOINT`, etc. See [.env.example](.env.example) and [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Contributing

Pull requests welcome. Ensure tests pass.

## License

MIT. See [LICENSE.md](LICENSE.md).

## Contact

[hi@rithul.dev](mailto:hi@rithul.dev)
