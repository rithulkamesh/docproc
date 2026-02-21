"""Embedding-based RAG using vector store + LLM."""

import uuid
from typing import List, Optional

from docproc.providers.base import ModelProvider
from docproc.rag.base import RAGBackend
from docproc.sanitize import sanitize_text, deduplicate_chunks
from docproc.stores.base import Chunk, VectorStore


class EmbeddingRAG(RAGBackend):
    """Traditional RAG: embed documents, store in vector DB, retrieve + generate."""

    def __init__(
        self,
        store: VectorStore,
        embed_provider: ModelProvider,
        llm_provider: ModelProvider,
        namespace: str = "default",
        chunk_size: int = 512,
        ingest_config=None,
    ):
        self.store = store
        self.embed_provider = embed_provider
        self.llm_provider = llm_provider
        self.namespace = namespace
        self.chunk_size = chunk_size
        self._ingest_config = ingest_config

    def index(
        self,
        documents: List[str],
        document_ids: List[str] | None = None,
        sanitize: Optional[bool] = None,
        dedupe: Optional[bool] = None,
    ) -> None:
        cfg = self._ingest_config
        do_sanitize = sanitize if sanitize is not None else (cfg.sanitize if cfg else True)
        do_dedupe = dedupe if dedupe is not None else (cfg.drop_exact_duplicates if cfg else True)
        doc_ids = document_ids or [str(uuid.uuid4()) for _ in documents]
        all_chunks: List[Chunk] = []
        for doc, doc_id in zip(documents, doc_ids):
            text = sanitize_text(doc) if do_sanitize else doc
            chunks = self._chunk(text)
            for i, c in enumerate(chunks):
                content = sanitize_text(c) if do_sanitize else c
                if not content:
                    continue
                all_chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        content=content,
                        metadata={"chunk_idx": i},
                    )
                )
        if do_dedupe:
            drop_bp = cfg.drop_boilerplate if cfg else True
            all_chunks = deduplicate_chunks(
                all_chunks,
                drop_exact_duplicates=True,
                drop_boilerplate=drop_bp,
            )
        texts_to_embed = [c.content for c in all_chunks]
        if texts_to_embed:
            embeddings = self.embed_provider.embed(texts_to_embed)
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.embedding = emb
            self.store.upsert(all_chunks, namespace=self.namespace)

    def _chunk(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        size = chunk_size or self.chunk_size
        words = text.split()
        chunks = []
        current = []
        for w in words:
            current.append(w)
            if len(" ".join(current)) >= size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks if chunks else [text]

    def query(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """Return (answer, sources) where sources is list of {content, document_id}."""
        q_embs = self.embed_provider.embed([question])
        results = self.store.search(
            q_embs[0], top_k=top_k, namespace=self.namespace
        )
        retrieved = [chunk.content for chunk, _ in results]
        sources = [
            {"content": chunk.content, "document_id": chunk.document_id}
            for chunk, _ in results
        ]
        context = "\n\n".join(retrieved)
        prompt = f"""Answer the question based only on the following context.

Context:
{context}

Question: {question}

Answer:"""
        from docproc.providers.base import ChatMessage
        resp = self.llm_provider.chat([ChatMessage(role="user", content=prompt)])
        return resp.content, sources
