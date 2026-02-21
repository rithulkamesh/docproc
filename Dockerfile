# DocProc v2 — Document Intelligence Platform
# Multi-stage build for a lean production image
# Build: docker build -t docproc:2.0 .
# Run:   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx docproc:2.0

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY docproc/ docproc/
COPY frontend/ frontend/

# Create venv and install with pip (self-contained, no path refs)
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime deps: tesseract for OCR, OpenCV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /build/docproc ./docproc
COPY --from=builder /build/frontend ./frontend
COPY --from=builder /build/README.md .
COPY docproc.example.yaml ./docproc.example.yaml
COPY docker/docproc.default.yaml ./docproc.yaml

LABEL org.opencontainers.image.title="DocProc"
LABEL org.opencontainers.image.description="Document Intelligence Platform"
LABEL org.opencontainers.image.source="https://github.com/rithulkamesh/docproc"
LABEL org.opencontainers.image.url="https://github.com/rithulkamesh/docproc/pkgs/container/docproc"

RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8000

ENV DOCPROC_CONFIG=/app/docproc.yaml

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/models')" || exit 1

CMD ["docproc-serve"]
