"""Open WebUI compatible OpenAI API endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str | List[dict] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "docproc-rag"
    messages: List[ChatMessage]
    stream: bool = False


def _get_rag():
    """Get RAG backend from config (single database + primary AI)."""
    from docproc.rag.factory import get_rag
    return get_rag()


@router.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list for Open WebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": "docproc-rag",
                "object": "model",
                "created": 0,
                "owned_by": "docproc",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions proxied to DocProc RAG."""
    rag = _get_rag()
    if rag is None:
        return {
            "id": "chatcmpl-docproc",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "RAG not configured."},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
    last_user = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user = m.content if isinstance(m.content, str) else str(m.content)
            break
    answer, _ = rag.query(last_user, top_k=5)
    return {
        "id": "chatcmpl-docproc",
        "object": "chat.completion",
        "choices": [
            {
                "message": {"role": "assistant", "content": answer},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
