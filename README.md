# docproc

**Document processor CLI** — File in, markdown out. High-accuracy extraction from PDF, DOCX, PPTX, XLSX using vision LLMs and optional LLM refinement. Multi-provider (OpenAI, Azure, Anthropic, Ollama, LiteLLM). Docproc is document processing only; assessment grading lives in the Go demo app.

**Full-stack demo (Go + React)** — The study workspace (upload, RAG chat, notes, flashcards, assessments) lives in **[demo/](demo/)**. It is a separate Go application that uses LocalStack (S3), RabbitMQ, PostgreSQL + PgVector, and invokes the docproc CLI only when a document is uploaded or when grading an answer.

---

## Features (CLI)

- **Extract** — `docproc --file input.pdf -o output.md`: native text + vision for every embedded image (equations, diagrams, labels).
- **Vision** — PDFs: native text layer; embedded images → Azure AI Vision or vision LLM (OpenAI, Anthropic, Ollama).
- **Refine** — Optional LLM pass: markdown, LaTeX math, boilerplate removed (`ingest.use_llm_refine`).
- **Config** — `docproc.yaml`: AI providers, ingest options; no server or database required for extract.

## Quick Start (CLI only)

```bash
git clone https://github.com/rithulkamesh/docproc.git && cd docproc
uv sync --python 3.12

# One-time: write ~/.config/docproc/docproc.yml from .env
uv run docproc init-config --env .env

# Extract a document to markdown
uv run docproc --file input.pdf -o output.md
```

## Demo (full stack)

See **[demo/README.md](demo/README.md)**. Run PostgreSQL, LocalStack, RabbitMQ via `docker compose`, then the Go API and worker; the React frontend in `demo/web/` talks to the Go app. Document processing is done by running the docproc CLI from the Go worker.

## Configuration

Create `docproc.yaml` (or use `docproc init-config` to generate from `.env`). For extract and grade, only AI and ingest matter:

```yaml
ai_providers:
  - provider: openai   # or azure, anthropic, ollama, litellm
primary_ai: openai

ingest:
  use_vision: true      # PDF: extract text + vision for images
  use_llm_refine: true   # Clean markdown, LaTeX, remove boilerplate
```

Secrets from environment or `.env`. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Installation

```bash
uv tool install git+https://github.com/rithulkamesh/docproc.git
# or: pip install git+https://github.com/rithulkamesh/docproc.git
```

From source: `uv sync --python 3.12` then `uv run docproc --file input.pdf -o output.md`.

## Usage

- **Extract:** `docproc --file input.pdf -o output.md` (optional `--config path`, `-v`).
- **Shell completions:** `docproc completions bash` or `docproc completions zsh`.

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Config schema, database options, AI providers, ingest, RAG |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline flow, modules, CLI vs API |
| [docs/AZURE_SETUP.md](docs/AZURE_SETUP.md) | Azure OpenAI + Azure AI Vision (Describe + Read), credentials |
| [docs/ASSESSMENTS_AI.md](docs/ASSESSMENTS_AI.md) | AI-generated assessments, grading pipeline, question types |

## Environment

- `DOCPROC_CONFIG` — Path to config file (default: `docproc.yaml`).
- Provider-specific: `OPENAI_API_KEY`, `AZURE_OPENAI_*`, `ANTHROPIC_API_KEY`, etc. See [.env.example](.env.example) and [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Contributing

Pull requests welcome. Ensure tests pass.

## License

MIT. See [LICENSE.md](LICENSE.md).

## Contact

[hi@rithul.dev](mailto:hi@rithul.dev)
