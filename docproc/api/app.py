"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docproc.api.routes import documents, query, models, openwebui


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocProc v2 API",
        description="Document Intelligence Platform",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(query.router, prefix="", tags=["query"])
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(openwebui.router, prefix="/api", tags=["openwebui"])
    return app
