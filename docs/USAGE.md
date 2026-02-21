# DocProc Usage Examples

## CLI

### Extract document to markdown

```bash
# With config
docproc --file input.pdf -o output.md --config docproc.yaml

# With DOCPROC_CONFIG env
export DOCPROC_CONFIG=docproc.yaml
docproc --file slides.pptx -o slides.md
```

### Supported formats

PDF, DOCX, PPTX, XLSX (same as API). Use `-o output.md` for markdown output.

See [docproc.cli.yaml](../docproc.cli.yaml) for an Ollama-only config example.

## API

### Start the server

```bash
DOCPROC_CONFIG=docproc.yaml docproc-serve
# API at http://localhost:8000
```

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@input.pdf"
# Returns: {"id": "...", "status": "processing"}
```

### List documents

```bash
curl http://localhost:8000/documents/
```

### Get document status and content

```bash
curl http://localhost:8000/documents/{document_id}
# Returns status, progress, full_text, regions when completed
```

### Query (RAG)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main idea?"}'
```

## Docker

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx ghcr.io/rithulkamesh/docproc:latest
```

See [README.md](../README.md) for full Docker Compose setup.

## Configuration

- [CONFIGURATION.md](CONFIGURATION.md) — Config schema, database and AI providers
- [AZURE_SETUP.md](AZURE_SETUP.md) — Azure OpenAI and Azure AI Vision setup
