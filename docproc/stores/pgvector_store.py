"""PgVector vector store implementation."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from docproc.stores.base import Chunk, VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector vector store."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "docproc_chunks",
    ):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost/docproc"
        )
        self.table_name = table_name
        self._engine = None
        self._Session = None
        self._init_db()

    def _init_db(self) -> None:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        self._engine = create_engine(self.connection_string)
        with self._engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception:
                pass
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255),
                        content TEXT,
                        metadata JSONB,
                        embedding vector(1536),
                        namespace VARCHAR(255),
                        page_ref INTEGER
                    )
                    """
                )
            )
            conn.commit()
        self._Session = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)

    def upsert(self, chunks: List[Chunk], namespace: Optional[str] = None) -> None:
        from sqlalchemy import text

        ns = namespace or "default"
        with self._Session() as session:
            for chunk in chunks:
                emb = chunk.embedding
                if emb is None:
                    emb = [0.0] * 1536
                meta_str = json.dumps(chunk.metadata) if isinstance(chunk.metadata, dict) else "{}"
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                session.execute(
                    text(
                        f"""
                        INSERT INTO {self.table_name} (id, document_id, content, metadata, embedding, namespace, page_ref)
                        VALUES (:id, :doc_id, :content, :metadata::jsonb, :embedding::vector, :ns, :page_ref)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "id": chunk.id,
                        "doc_id": chunk.document_id,
                        "content": chunk.content,
                        "metadata": meta_str,
                        "embedding": emb_str,
                        "ns": ns,
                        "page_ref": chunk.page_ref,
                    },
                )
            session.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        from sqlalchemy import text

        ns = namespace or "default"
        emb_str = str(query_embedding)
        with self._Session() as session:
            result = session.execute(
                text(
                    f"""
                    SELECT id, document_id, content, metadata, page_ref,
                           1 - (embedding <=> :emb::vector) as score
                    FROM {self.table_name}
                    WHERE namespace = :ns
                    ORDER BY embedding <=> :emb::vector
                    LIMIT :top_k
                    """
                ),
                {"emb": emb_str, "ns": ns, "top_k": top_k},
            )
            rows = result.fetchall()
        out = []
        for row in rows:
            chunk = Chunk(
                id=row[0],
                document_id=row[1],
                content=row[2],
                metadata=row[3] or {},
                page_ref=row[4],
            )
            out.append((chunk, float(row[5])))
        return out

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        from sqlalchemy import text

        ns = namespace or "default"
        with self._Session() as session:
            for i in ids:
                session.execute(
                    text(
                        f"DELETE FROM {self.table_name} WHERE id = :id AND namespace = :ns"
                    ),
                    {"id": i, "ns": ns},
                )
            session.commit()

    def get_by_id(self, chunk_id: str, namespace: Optional[str] = None) -> Optional[Chunk]:
        from sqlalchemy import text

        ns = namespace or "default"
        with self._Session() as session:
            result = session.execute(
                text(
                    f"SELECT id, document_id, content, metadata, page_ref FROM {self.table_name} WHERE id = :id AND namespace = :ns"
                ),
                {"id": chunk_id, "ns": ns},
            )
            row = result.fetchone()
        if row is None:
            return None
        return Chunk(
            id=row[0],
            document_id=row[1],
            content=row[2],
            metadata=row[3] or {},
            page_ref=row[4],
        )
