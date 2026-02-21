# DocProc Configuration Guide

DocProc uses a single configuration file as the **source of truth**. You choose **one database provider** and can configure **multiple AI providers**; one AI provider is primary for RAG, vision extraction, and LLM refinement.

See [README.md](README.md) for the documentation index.

## Configuration File

Create `docproc.yaml` in the project root, or set `DOCPROC_CONFIG` to the path of your config file.

**Search order:** explicit path → `DOCPROC_CONFIG` env → `docproc.yaml` → `docproc.yml` → `~/.docproc.yaml`

## Schema

### Database (Single Provider)

Only **one** database is used. Choose one of: `pgvector`, `qdrant`, `chroma`, `faiss`, `memory`.

```yaml
database:
  provider: qdrant  # required
  connection_string: http://localhost:6333
  collection_name: docproc
  table_name: docproc_chunks  # pgvector only
  embed_dim: 1536
  path: ./data  # for chroma/faiss persistent dir
```

| Provider  | connection_string / path        | Notes                          |
|-----------|---------------------------------|--------------------------------|
| pgvector  | `postgresql://host/db`          | Needs pgvector extension       |
| qdrant    | `http://localhost:6333`         | Default                        |
| chroma    | directory path                  | Persists to disk               |
| faiss     | directory path                  | Persists index + metadata      |
| memory    | (ignored)                       | In-memory, ephemeral           |

### AI Providers (Multiple Allowed)

```yaml
ai_providers:
  - provider: openai
    default_model: gpt-4o
    default_vision_model: gpt-4o
  - provider: anthropic
    default_model: claude-sonnet-4-20250514
  - provider: ollama
    base_url: http://localhost:11434
    default_model: llava
  - provider: litellm
    default_model: gpt-4o

primary_ai: openai  # which provider to use for RAG/embed
```

| Provider   | Config keys          | Env vars                          |
|------------|----------------------|-----------------------------------|
| openai     | api_key, base_url    | OPENAI_API_KEY, OPENAI_BASE_URL   |
| azure      | api_key, base_url    | AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_VISION_ENDPOINT |
| anthropic  | api_key              | ANTHROPIC_API_KEY                 |
| ollama     | base_url             | OLLAMA_BASE_URL                   |
| litellm    | default_model        | Uses LiteLLM model strings        |

### Ingest (PDF vision extraction)

```yaml
ingest:
  sanitize: true
  drop_exact_duplicates: true
  drop_boilerplate: true
  use_vision: true  # PDF: send embedded images to vision LLM; false = text only
  use_llm_refine: true  # LLM pass: markdown, LaTeX math, remove boilerplate before indexing
```

- `use_vision`: When true (default), PDFs use text layer + vision extraction for embedded images. When false, only native PDF text is extracted (faster, no vision API calls).
- `use_llm_refine`: When true (default), extracted text is refined via LLM: boilerplate removed, equations → LaTeX, figure JSON artifacts cleaned, output as markdown. Applied before RAG indexing.

### Azure AI Vision (image extraction)

If you have **Azure AI Vision (Computer Vision)** deployed but no Azure OpenAI chat deployment, set:

```
AZURE_VISION_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
```

Uses the same key as `AZURE_OPENAI_API_KEY` when both are in the same resource. Image extraction will use the Computer Vision Describe API instead of chat completions.

### RAG

```yaml
rag:
  backend: embedding  # embedding | clara
  top_k: 5
  chunk_size: 512
  namespace: default
```

## Upload and progress

- `POST /documents/upload` returns immediately with `{ "id": "<uuid>", "status": "processing" }`. Processing runs in the background.
- Poll `GET /documents/{id}` for `status` (`processing` | `completed` | `failed`) and optional `progress: { "page", "total", "message" }`.
- The Streamlit frontend uses this to show a per-file progress bar.

## Docker

The image includes a default config (`/app/docproc.yaml`) using the `memory` database. Override by:
- Mounting a config: `-v ./docproc.yaml:/app/docproc.yaml`
- Setting `DOCPROC_CONFIG` to the mounted path

## Environment

| Variable | Purpose |
|----------|---------|
| `DOCPROC_CONFIG` | Config file path (default: `docproc.yaml`) |
| `DOCPROC_API_URL` | API base URL for Streamlit frontend (default: `http://localhost:8000`) |
| `DOCPROC_DATABASE_PROVIDER` | Override `database.provider` |
| `DATABASE_URL` | Override database connection string (e.g. Postgres) |
| `OPENAI_API_KEY` | OpenAI / Azure key when using OpenAI or Azure provider |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI deployment |
| `AZURE_VISION_ENDPOINT` | Azure AI Vision (Computer Vision) for image extraction |
| `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, etc. | Per-provider; see schema and [.env.example](../.env.example) |
