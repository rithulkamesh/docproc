"""FAISS vector store implementation."""

import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docproc.stores.base import Chunk, VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    """FAISS vector store. Saves index + metadata to disk."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        embed_dim: int = 1536,
    ):
        """Initialize FAISS store.

        Args:
            index_path: Directory to persist index and metadata
            embed_dim: Embedding dimension
        """
        self.index_path = Path(index_path) if index_path else None
        self.embed_dim = embed_dim
        self._index = None
        self._id_to_chunk: Dict[str, Chunk] = {}
        self._id_list: List[str] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        import faiss

        if self.index_path and self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path / "index.faiss"))
            meta_path = self.index_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    data = json.load(f)
                self._id_list = data.get("ids", [])
                chunks_data = data.get("chunks", {})
                for cid, c in chunks_data.items():
                    self._id_to_chunk[cid] = Chunk(
                        id=c["id"],
                        document_id=c.get("document_id", ""),
                        content=c.get("content", ""),
                        metadata=c.get("metadata", {}),
                        page_ref=c.get("page_ref"),
                    )
        else:
            self._index = faiss.IndexFlatIP(self.embed_dim)
            self._id_list = []
            self._id_to_chunk = {}

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current chunks (IndexFlatIP has no in-place updates)."""
        import faiss
        import numpy as np

        vectors = []
        new_id_list = []
        for cid in self._id_list:
            c = self._id_to_chunk.get(cid)
            if not c or not getattr(c, "embedding", None):
                continue
            vectors.append(c.embedding)
            new_id_list.append(cid)
        self._id_list = new_id_list
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(arr)
            self._index = faiss.IndexFlatIP(self.embed_dim)
            self._index.add(arr)
        else:
            self._index = faiss.IndexFlatIP(self.embed_dim)

    def _save(self) -> None:
        if not self.index_path:
            return
        self.index_path.mkdir(parents=True, exist_ok=True)
        import faiss
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        chunks_data = {
            cid: {
                "id": c.id,
                "document_id": c.document_id,
                "content": c.content,
                "metadata": c.metadata or {},
                "page_ref": c.page_ref,
            }
            for cid, c in self._id_to_chunk.items()
        }
        with open(self.index_path / "metadata.json", "w") as f:
            json.dump({"ids": self._id_list, "chunks": chunks_data}, f, indent=2)

    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        for chunk in chunks:
            emb = chunk.embedding
            if emb is None:
                emb = [0.0] * self.embed_dim
            c = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=dict(chunk.metadata or {}, namespace=ns),
                page_ref=chunk.page_ref,
                embedding=emb,
            )
            self._id_to_chunk[chunk.id] = c
            if chunk.id not in self._id_list:
                self._id_list.append(chunk.id)
        self._rebuild_index()
        self._save()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        import numpy as np

        ns = namespace or "default"
        if self._index.ntotal == 0:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) or 1e-9)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        out = []
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            cid = self._id_list[idx]
            c = self._id_to_chunk.get(cid)
            if c and (c.metadata or {}).get("namespace") == ns:
                out.append((c, float(s)))
        return out[:top_k]

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        for cid in ids:
            self._id_to_chunk.pop(cid, None)
            if cid in self._id_list:
                self._id_list.remove(cid)
        self._rebuild_index()
        self._save()

    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        return self._id_to_chunk.get(chunk_id)
