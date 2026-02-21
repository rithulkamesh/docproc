"""Base vector store interface and document schemas.

Single database provider per deployment. Implementations: PgVector, Qdrant,
Chroma, FAISS, Memory. Use stores.factory.create_store() for config-driven access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Document:
    """Document with metadata. Used for ingestion before chunking."""

    id: str
    content: str
    metadata: Dict[str, Any]
    source: Optional[str] = None


@dataclass
class Chunk:
    """Document chunk with optional embedding. Stored in vector DB."""

    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    page_ref: Optional[int] = None


class VectorStore(ABC):
    """Abstract base for vector stores. One store per DocProc instance."""

    @abstractmethod
    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        """Insert or update chunks. Embeddings must be set for semantic search."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search by embedding. Returns (chunk, score) pairs sorted by relevance."""
        pass

    @abstractmethod
    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        """Delete chunks by ID."""
        pass

    @abstractmethod
    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        """Fetch a single chunk by ID."""
        pass
