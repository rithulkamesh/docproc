"""Vector store providers — single database, config-driven."""

from docproc.stores.base import VectorStore, Document, Chunk
from docproc.stores.factory import create_store
from docproc.stores.pgvector_store import PgVectorStore
from docproc.stores.qdrant_store import QdrantStore
from docproc.stores.chroma_store import ChromaStore
from docproc.stores.faiss_store import FAISSStore
from docproc.stores.memory_store import MemoryStore

__all__ = [
    "VectorStore",
    "Document",
    "Chunk",
    "create_store",
    "PgVectorStore",
    "QdrantStore",
    "ChromaStore",
    "FAISSStore",
    "MemoryStore",
]
