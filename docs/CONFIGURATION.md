# DocProc Configuration Guide

DocProc uses a single configuration file as the **source of truth**. You choose **one database provider** and can configure **multiple AI providers**; one AI provider is primary for RAG and vision extraction.

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
| azure      | api_key, base_url    | AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT |
| anthropic  | api_key              | ANTHROPIC_API_KEY                 |
| ollama     | base_url             | OLLAMA_BASE_URL                   |
| litellm    | default_model        | Uses LiteLLM model strings        |

### RAG

```yaml
rag:
  backend: embedding  # embedding | clara
  top_k: 5
  chunk_size: 512
  namespace: default
```

## Docker

The image includes a default config (`/app/docproc.yaml`) using the `memory` database. Override by:
- Mounting a config: `-v ./docproc.yaml:/app/docproc.yaml`
- Setting `DOCPROC_CONFIG` to the mounted path

## Environment Overrides

- `DOCPROC_CONFIG` — config file path
- `DOCPROC_DATABASE_PROVIDER` — override database.provider
- `DATABASE_URL` — override connection string
- `QDRANT_URL`, `OPENAI_API_KEY`, etc. — per-provider
