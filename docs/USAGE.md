# docproc Usage Examples

## CLI

### Extract document to markdown

```bash
docproc --file input.pdf -o output.md

# With explicit config
docproc --file input.pdf -o output.md --config docproc.yaml

# Verbose
docproc --file slides.pptx -o slides.md -v
```

### Supported formats

PDF, DOCX, PPTX, XLSX. Output must be a `.md` file (`-o output.md`).

### One-time config from .env

```bash
docproc init-config --env .env
# Writes ~/.config/docproc/docproc.yml from your .env (AI keys, etc.)
```

### Shell completions

```bash
docproc completions bash   # or zsh
# Source the output in your shell to get completions for --file, -o, --config
```

## Configuration

- [CONFIGURATION.md](CONFIGURATION.md) — Config schema, AI providers, ingest options
- [AZURE_SETUP.md](AZURE_SETUP.md) — Azure OpenAI and Azure AI Vision setup

## Full-stack demo

For the document workspace (upload, RAG chat, notes, assessments), see **[demo/README.md](../demo/README.md)**. The demo uses Docker Compose for infrastructure (PostgreSQL, LocalStack, RabbitMQ) and runs the Go API and React frontend on the host. Document processing is done by the docproc CLI invoked from the Go worker.
