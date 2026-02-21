"""Base RAG backend interface."""

from abc import ABC, abstractmethod
from typing import List


class RAGBackend(ABC):
    """Abstract base for RAG backends (CLaRa, embedding-based)."""

    @abstractmethod
    def index(self, documents: List[str], document_ids: List[str] | None = None) -> None:
        """Index documents for retrieval."""
        pass

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> tuple[str, List[str]]:
        """Query and return (answer, retrieved_doc_contents)."""
        pass
