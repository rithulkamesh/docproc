# DocProc Documentation

## Guides

| Document | Description |
|----------|-------------|
| [CONFIGURATION.md](CONFIGURATION.md) | **Configuration reference** — `docproc.yaml` schema, database providers (PgVector, Qdrant, Chroma, FAISS, memory), AI providers (OpenAI, Azure, Anthropic, Ollama, LiteLLM), ingest options (vision, LLM refinement), RAG, environment overrides |
| [AZURE_SETUP.md](AZURE_SETUP.md) | **Azure setup** — Azure OpenAI deployments, Azure AI Vision (Computer Vision) for image extraction (Describe + Read API), credentials via env or `scripts/azure_env.sh` |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Architecture overview** — Pipeline flow, modules, CLI vs API |
| [USAGE.md](USAGE.md) | **Usage examples** — CLI, API, Docker, curl examples |

See also [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and running tests.

## Concepts

- **Single store** — One vector database at a time; chosen in `database.provider`.
- **Primary AI** — One provider is used for RAG and (when enabled) for vision extraction and LLM refinement; set in `primary_ai`.
- **Ingest pipeline** — Extract (native text + vision for images) → optional LLM refine (markdown, LaTeX) → sanitize/dedupe → index.
- **Progress** — Upload returns immediately; `GET /documents/{id}` includes `progress: { page, total, message }` until `status` is `completed` or `failed`.
