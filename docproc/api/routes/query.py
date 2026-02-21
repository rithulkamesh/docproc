"""RAG query endpoint. Uses config-driven single database + primary AI provider."""

from fastapi import APIRouter
from pydantic import BaseModel

from docproc.rag.factory import get_rag

router = APIRouter()

# Import documents for enriching sources with filename
from docproc.api.routes import documents as docs_route


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
            "sources": [],
        }
    answer, raw = rag.query(req.prompt, top_k=req.top_k)
    # Normalize: EmbeddingRAG returns list[dict], CLaRa returns list[str]
    if raw and isinstance(raw[0], dict):
        sources = raw
    else:
        sources = [{"content": c, "document_id": ""} for c in (raw or [])]
    for s in sources:
        doc_id = s.get("document_id", "")
        doc = docs_route._documents.get(doc_id, {})
        s["filename"] = doc.get("filename", "")
    return {"answer": answer, "sources": sources}
