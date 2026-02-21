"""RAG query endpoint. Uses config-driven single database + primary AI provider."""

from fastapi import APIRouter
from pydantic import BaseModel

from docproc.rag.factory import get_rag

router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for RAG query."""

    prompt: str
    top_k: int = 5


@router.post("/query")
async def query(req: QueryRequest):
    """Run RAG query. Backend and providers come from config (docproc.yaml)."""
    rag = get_rag()
    if rag is None:
        return {
            "answer": "RAG not configured. Create docproc.yaml and set database + ai_providers.",
            "retrieved": [],
        }
    answer, retrieved = rag.query(req.prompt, top_k=req.top_k)
    return {"answer": answer, "retrieved": retrieved}
