"""Qdrant vector store implementation."""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from docproc.stores.base import Chunk, VectorStore


class QdrantStore(VectorStore):
    """Qdrant vector store."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: str = "docproc",
        embed_dim: int = 1536,
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=self.url)
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            )

    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        from qdrant_client.models import PointStruct

        ns = namespace or "default"
        points = []
        for chunk in chunks:
            emb = chunk.embedding
            if emb is None:
                emb = [0.0] * self.embed_dim
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["namespace"] = ns
            meta["document_id"] = chunk.document_id
            meta["page_ref"] = chunk.page_ref
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=emb,
                    payload={
                        "content": chunk.content,
                        "metadata": meta,
                        "document_id": chunk.document_id,
                        "namespace": ns,
                        "page_ref": chunk.page_ref,
                    },
                )
            )
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        ns = namespace or "default"
        query_filter = Filter(
            must=[FieldCondition(key="namespace", match=MatchValue(value=ns))]
        )
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        out = []
        for hit in results:
            payload = hit.payload
            chunk = Chunk(
                id=str(hit.id),
                document_id=payload.get("document_id", ""),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
                page_ref=payload.get("page_ref"),
            )
            out.append((chunk, hit.score or 0.0))
        return out

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id],
        )
        if not results:
            return None
        payload = results[0].payload
        return Chunk(
            id=str(results[0].id),
            document_id=payload.get("document_id", ""),
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
            page_ref=payload.get("page_ref"),
        )
