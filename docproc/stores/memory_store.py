"""In-memory vector store (ephemeral, for testing)."""

from typing import Any, Dict, List, Optional, Tuple

from docproc.stores.base import Chunk, VectorStore


class MemoryStore(VectorStore):
    """In-memory vector store. Data is lost when the process exits."""

    def __init__(self, embed_dim: int = 1536):
        self.embed_dim = embed_dim
        self._chunks: Dict[str, Tuple[Chunk, List[float]]] = {}

    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        for chunk in chunks:
            emb = chunk.embedding or [0.0] * self.embed_dim
            meta = dict(chunk.metadata or {}, namespace=ns)
            c = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=meta,
                page_ref=chunk.page_ref,
            )
            self._chunks[chunk.id] = (c, emb)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        ns = namespace or "default"
        results = []
        for chunk, emb in self._chunks.values():
            if (chunk.metadata or {}).get("namespace") != ns:
                continue
            score = _cosine_sim(query_embedding, emb)
            results.append((chunk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        for i in ids:
            self._chunks.pop(i, None)

    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        t = self._chunks.get(chunk_id)
        return t[0] if t else None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)
