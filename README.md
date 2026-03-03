# docproc

docproc turns documents into markdown. Give it a PDF, DOCX, PPTX, or XLSX; you get clean text and every image (equations, diagrams, labels) explained by a vision model. It’s CLI only. Works with OpenAI, Azure, Anthropic, Ollama, or LiteLLM.

The **docproc // edu** demo in [demo/](demo/) is a full study workspace: upload docs, chat over them, generate notes and flashcards, create and take assessments. That app is written in Go and calls this CLI when a document is uploaded; it does grading itself.

---

## What the CLI does

**Extract.** `docproc --file input.pdf -o output.md` — Pulls text from the native layer and runs vision on every embedded image. Optional extra pass: tidy markdown, LaTeX math, strip boilerplate (see `ingest.use_llm_refine` in config).

**Config.** `docproc.yaml` holds AI providers and ingest options. No database or server needed for extract. Use `docproc init-config --env .env` once to generate a starter config from your `.env`.

## Quick start

```bash
git clone https://github.com/rithulkamesh/docproc.git && cd docproc
uv sync --python 3.12

uv run docproc init-config --env .env   # one-time
uv run docproc --file input.pdf -o output.md
```

## Demo (docproc // edu)

See [demo/README.md](demo/README.md). From `demo/`, run `docker compose up -d` (stack name: **docproc-edu**). Then start the Go API and worker from `demo/go/`, and the React app from `demo/web/`. The worker runs the docproc CLI on each uploaded document.

## Configuration

Create `docproc.yaml` or generate from `.env` with `init-config`. For both the CLI and the demo, the bits that matter are AI providers and ingest:

```yaml
ai_providers:
  - provider: openai   # or azure, anthropic, ollama, litellm
primary_ai: openai

ingest:
  use_vision: true
  use_llm_refine: true
```

Secrets go in the environment or `.env`. Full schema: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Install

```bash
uv tool install git+https://github.com/rithulkamesh/docproc.git
# or: pip install git+https://github.com/rithulkamesh/docproc.git
```

From source: `uv sync --python 3.12` then `uv run docproc --file input.pdf -o output.md`.

## Usage

- **Extract:** `docproc --file input.pdf -o output.md` (optional `--config path`, `-v`).
- **Completions:** `docproc completions bash` or `docproc completions zsh`.

## Docs

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Index |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Config schema, providers, ingest, RAG |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, CLI vs API |
| [docs/AZURE_SETUP.md](docs/AZURE_SETUP.md) | Azure OpenAI and Vision setup |
| [docs/ASSESSMENTS_AI.md](docs/ASSESSMENTS_AI.md) | Assessments and grading in the demo |

**Environment:** `DOCPROC_CONFIG` for config path (default: `docproc.yaml`). Provider keys: `OPENAI_API_KEY`, `AZURE_OPENAI_*`, `ANTHROPIC_API_KEY`, etc. See [.env.example](.env.example).

## Contributing

Pull requests welcome. Run the tests before sending.

## License

MIT. See [LICENSE.md](LICENSE.md).

---

## Why I built this

I learn by asking questions. Not surface-level ones—the deep "why"s that most materials never answer. When my peers studied from slides and PDFs, I got stuck. I couldn’t absorb content I wasn’t allowed to interrogate. Documents don’t talk back. They don’t explain the intuition or the connections. Tools like NotebookLM didn’t help: they don’t understand images in the source, so those parts showed up blank. Most of my slides were visual or screenshots. I had nothing to work with.

So I built something for myself. A way to pull content out of any document—slides, papers, textbooks—and ask AI the questions I needed. *Why does this work? What’s the reasoning here? How does this connect to what we did last week?* It grew from "extract and query" into a full study environment: chat over the corpus, generate notes and flashcards, create and take assessments with automatic grading. For the first time I could learn from static documents by *conversing*, *noting*, and *testing*—not just re-reading.

I’m open-sourcing it because I’m probably not the only one who learns this way.

[hi@rithul.dev](mailto:hi@rithul.dev)
