# Running the demo with Docker

The **demo** uses Docker Compose only for infrastructure. The Go API and React frontend run on the host (see [demo/README.md](../demo/README.md)).

## Demo infrastructure (PostgreSQL, LocalStack, RabbitMQ)

From the repo root:

```bash
cd demo && docker compose up -d
```

This starts:

| Service    | Port  | Purpose                          |
|-----------|-------|----------------------------------|
| postgres  | 5432  | PostgreSQL + pgvector (documents, RAG chunks, assessments) |
| localstack| 4566  | S3-compatible storage (document uploads) |
| rabbitmq  | 5672  | AMQP (document processing jobs); management UI on 15672 |

**Then run on the host:**

- Go API: `cd demo/go && go run .` (default port 8080)
- Worker: `cd demo/go && go run . --worker`
- Frontend: `cd demo/web && npm install && npm run dev` (default port 3000)

Ensure **docproc** CLI is on your PATH for the worker (e.g. `pip install -e .` from repo root). Set `OPENAI_API_KEY` (and optionally `DATABASE_URL`, `S3_ENDPOINT`, `MQ_URL`) for the Go app.

See [demo/README.md](../demo/README.md) for full quick start and environment variables.
