"""Chroma vector store implementation."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from docproc.stores.base import Chunk, VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Chroma vector store. Persistent or ephemeral."""

    def __init__(
        self,
        collection_name: str = "docproc",
        persist_directory: Optional[str] = None,
        embed_dim: int = 1536,
    ):
        """Initialize Chroma store.

        Args:
            collection_name: Collection name
            persist_directory: Directory for persistence; None = in-memory
            embed_dim: Embedding dimension
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embed_dim = embed_dim
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        import chromadb

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        for chunk in chunks:
            emb = chunk.embedding
            if emb is None:
                emb = [0.0] * self.embed_dim
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append({
                "document_id": chunk.document_id,
                "namespace": ns,
                "page_ref": chunk.page_ref or -1,
                **(chunk.metadata or {}),
            })
            embeddings.append(emb)
        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        ns = namespace or "default"
        where = {"namespace": ns}
        if filter_metadata:
            where.update(filter_metadata)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )
        out = []
        if results and results["ids"] and results["ids"][0]:
            for i, rid in enumerate(results["ids"][0]):
                meta = (results.get("metadatas") or [[]])[0][i] if i < len((results.get("metadatas") or [[]])[0]) else {}
                doc = (results.get("documents") or [[]])[0][i] if results.get("documents") and i < len(results["documents"][0]) else ""
                score = 1.0 - (results.get("distances") or [[]])[0][i] if results.get("distances") else 0.0
                chunk = Chunk(
                    id=rid,
                    document_id=meta.get("document_id", ""),
                    content=doc,
                    metadata={k: v for k, v in (meta or {}).items() if k not in ("document_id", "namespace", "page_ref")},
                    page_ref=meta.get("page_ref") if meta.get("page_ref") != -1 else None,
                )
                out.append((chunk, float(score)))
        return out

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        self._collection.delete(ids=ids)

    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        results = self._collection.get(ids=[chunk_id])
        ids = results.get("ids") if results else []
        if not ids:
            return None
        meta = (results.get("metadatas") or [{}])[0]
        doc = (results.get("documents") or [""])[0]
        m = meta if isinstance(meta, dict) else {}
        return Chunk(
            id=chunk_id,
            document_id=m.get("document_id", ""),
            content=doc,
            metadata={k: v for k, v in m.items() if k not in ("document_id", "namespace", "page_ref")},
            page_ref=m.get("page_ref") if m.get("page_ref") != -1 else None,
        )
