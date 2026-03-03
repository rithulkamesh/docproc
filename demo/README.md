# docproc // edu

Full-stack demo: Go API + React frontend. Document processing is done by the **docproc** CLI (Python). This app handles uploads, storage (LocalStack S3), message queue (RabbitMQ), RAG (PgVector), and **assessment grading** (single-select, formula, conceptual, derivation) in Go.

## Prerequisites

- Go 1.22+
- Node.js (for `demo/web/` frontend)
- Docker (for PostgreSQL, LocalStack, RabbitMQ)
- docproc CLI installed (`pip install -e .` from repo root)

## Quick start

1. Start the full stack (API, worker with docproc CLI, web, Postgres, LocalStack, RabbitMQ):
   ```bash
   cd demo && cp .env.example .env   # then edit .env with Azure (or OpenAI) creds
   docker compose up -d
   ```
   Compose loads `demo/.env` and passes Azure/OpenAI vars into the api and worker. For Azure, set `DOCPROC_PRIMARY_AI=azure` and the `AZURE_OPENAI_*` (and optional `AZURE_VISION_*`) vars in `.env`.

2. Create S3 bucket (LocalStack) if not already created:
   ```bash
   aws --endpoint-url=http://localhost:4566 s3 mb s3://docproc-demo 2>/dev/null || true
   ```

3. Or run API + frontend locally (with infra in Docker):
   - API: `cd go && go run .`
   - Frontend: `cd web && npm install && npm run dev`
   - Open http://localhost:3000 (API defaults to http://localhost:8080).

## Worker (document processing)

- **With Docker:** The `worker` service image includes the docproc CLI. Uploads are processed automatically. Set `OPENAI_API_KEY` (e.g. in `demo/.env`) for vision/LLM extraction.
- **Locally:** Run a worker that uses your local docproc CLI:
  ```bash
  cd go && go run . --worker
  ```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | HTTP server port |
| DATABASE_URL | postgresql://docproc:docproc@localhost:5432/docproc?sslmode=disable | PostgreSQL |
| S3_ENDPOINT | http://localhost:4566 | LocalStack S3 |
| S3_BUCKET | docproc-demo | Bucket name |
| MQ_URL | amqp://docproc:docproc@localhost:5672/ | RabbitMQ |
| DOCPROC_CLI | docproc | Path to docproc binary |
| OPENAI_API_KEY | (required for RAG + grading) | Embeddings, RAG, and assessment grading |
