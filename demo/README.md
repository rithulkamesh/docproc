# docproc // edu

Full-stack demo: Go API + React frontend. Document processing is done by the **docproc** CLI (Python). This app handles uploads, storage (LocalStack S3), message queue (RabbitMQ), RAG (PgVector), and **assessment grading** (single-select, formula, conceptual, derivation) in Go.

## Prerequisites

- Go 1.22+
- Node.js (for `demo/web/` frontend)
- Docker (for PostgreSQL, LocalStack, RabbitMQ)
- docproc CLI installed (`pip install -e .` from repo root)

## Quick start

1. **Dev (recommended):** infra in Docker, API + web run locally so you see logs and avoid rebuilds.
   ```bash
   cd demo && cp .env.example .env   # edit .env with Azure (or OpenAI) creds
   just deps   # starts postgres, rabbitmq, localstack only
   just dev    # runs API + web in one terminal; Ctrl+C kills both, logs from both
   ```
   Open http://localhost:5173 (Vite). API at http://localhost:8080.

2. **Create S3 bucket** (LocalStack) if not already created:
   ```bash
   aws --endpoint-url=http://localhost:4566 s3 mb s3://docproc-demo 2>/dev/null || true
   ```

3. **Full stack in Docker** (prod-like; rebuilds on image change):
   ```bash
   cd demo && docker compose -f docker-compose-prod.yml up -d
   ```
   Web: http://localhost:3000, API: http://localhost:8080.

## Compose files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | **Dev:** postgres, rabbitmq, localstack only. Used by `just deps` and `just dev`. |
| `docker-compose-prod.yml` | **Prod/full:** infra + api + worker + web. Use when you want everything in containers. |

## Just commands

From `demo/` ([just](https://github.com/casey/just)):

| Command | What it does |
|---------|----------------|
| `just deps` | Start dev services (postgres, rabbitmq, localstack). |
| `just dev` | Start deps, then run API + web in one terminal; logs from both; **Ctrl+C kills both**. |
| `just down` | Stop dev services (docker compose down). |
| `just api` / `just web` | Run only API or only web (for two-terminal workflow). |
| `just api-watch` | API with live reload (requires `air`: `go install github.com/air-verse/air@latest`). |
| `just prod-up` | Full stack via docker-compose-prod.yml. |
| `just prod-down` | Stop full stack. |
| `just api-restart` / `just web-restart` | Rebuild and restart that container (prod compose). |

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
