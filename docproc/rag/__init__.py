"""RAG backends. Use factory.get_rag() for config-driven access."""

from docproc.rag.base import RAGBackend
from docproc.rag.factory import create_rag, get_rag
from docproc.rag.embedding_rag import EmbeddingRAG
from docproc.rag.clara_rag import ClaraRAG

__all__ = ["RAGBackend", "create_rag", "get_rag", "EmbeddingRAG", "ClaraRAG"]
