# docproc Documentation

## Guides

| Document | Description |
|----------|-------------|
| [CONFIGURATION.md](CONFIGURATION.md) | **Configuration reference** — `docproc.yaml` schema, AI providers (OpenAI, Azure, Anthropic, Ollama, LiteLLM), ingest options (vision, LLM refinement). Used by the CLI for document extraction. |
| [AZURE_SETUP.md](AZURE_SETUP.md) | **Azure setup** — Azure OpenAI and Azure AI Vision (Computer Vision) for PDF image extraction; credentials via env or `scripts/azure_env.sh`. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Architecture overview** — Pipeline flow, modules. docproc is CLI-only (file in → markdown out); the full-stack demo lives in `demo/`. |
| [USAGE.md](USAGE.md) | **Usage examples** — CLI extract, init-config, completions. |
| [DOCKER.md](DOCKER.md) | **Docker** — Demo infrastructure only (PostgreSQL, LocalStack, RabbitMQ). Go API and frontend run on the host. |

See also [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and running tests.

## Concepts

- **docproc (CLI)** — Document processor only. Reads a file (PDF, DOCX, PPTX, XLSX), extracts text (native + optional vision for images), optionally refines with an LLM, and writes markdown to a file. No server, no database, no RAG.
- **Demo** — Separate application in `demo/`: Go API, React frontend (`demo/web/`), document upload (LocalStack S3), job queue (RabbitMQ), RAG and grading (PostgreSQL + PgVector, OpenAI). Document processing is done by running the docproc CLI from the Go worker. See [demo/README.md](../demo/README.md).
