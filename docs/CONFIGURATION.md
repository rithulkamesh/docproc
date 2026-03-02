# docproc Configuration Guide

docproc (CLI) uses a single configuration file for **document extraction**: AI providers and **ingest** options (vision, LLM refinement). No database or RAG config is required for the CLI.

See [README.md](README.md) for the documentation index.

## Configuration file

Create `docproc.yaml` in the project root, or set `DOCPROC_CONFIG` to the path of your config file.

**Search order:** explicit path → `DOCPROC_CONFIG` env → `docproc.yaml` → `docproc.yml` → `~/.config/docproc/docproc.yml` → `~/.docproc.yaml`

You can also generate a starter config with `docproc init-config --env .env`, which writes `~/.config/docproc/docproc.yml`.

## Schema (CLI-relevant)

### AI providers

The CLI uses one primary AI provider for vision extraction and LLM refinement.

```yaml
ai_providers:
  - provider: openai
    default_model: gpt-4o
    default_vision_model: gpt-4o
  - provider: anthropic
    default_model: claude-sonnet-4-20250514
  - provider: ollama
    base_url: http://localhost:11434
    default_vision_model: llava

primary_ai: openai
```

| Provider   | Config keys          | Env vars                          |
|------------|----------------------|-----------------------------------|
| openai     | api_key, base_url    | OPENAI_API_KEY, OPENAI_BASE_URL   |
| azure      | api_key, base_url    | AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_VISION_ENDPOINT |
| anthropic  | api_key              | ANTHROPIC_API_KEY                 |
| ollama     | base_url             | OLLAMA_BASE_URL                   |
| litellm    | default_model        | Uses LiteLLM model strings        |

### Ingest (extraction behavior)

```yaml
ingest:
  sanitize: true
  drop_exact_duplicates: true
  drop_boilerplate: true
  use_vision: true   # PDF: send embedded images to vision LLM; false = text only
  use_llm_refine: true  # LLM pass: markdown, LaTeX, remove boilerplate
```

- **use_vision:** When true (default), PDFs use the native text layer plus vision extraction for embedded images. When false, only native PDF text is extracted (faster, no vision API calls).
- **use_llm_refine:** When true (default), extracted text is refined via LLM before writing the .md file.

### Azure AI Vision (image extraction)

If you have **Azure AI Vision (Computer Vision)** deployed:

Set `AZURE_VISION_ENDPOINT` in the environment. Image extraction will use the Computer Vision Describe + Read APIs. See [AZURE_SETUP.md](AZURE_SETUP.md).

## Environment

| Variable | Purpose |
|----------|---------|
| `DOCPROC_CONFIG` | Config file path (default: search order above) |
| `OPENAI_API_KEY` | OpenAI (or Azure key when using Azure provider) |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI |
| `AZURE_VISION_ENDPOINT` | Azure AI Vision for image extraction |
| `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, etc. | Per-provider; see [.env.example](../.env.example) |

**Note:** The **demo** (Go app) has its own configuration (e.g. `DATABASE_URL`, `S3_ENDPOINT`, `MQ_URL`, `OPENAI_API_KEY`). See [demo/README.md](../demo/README.md).
