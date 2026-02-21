"""Apple CLaRa-based RAG with semantic compression."""

import logging
from typing import List, Optional

from docproc.rag.base import RAGBackend

logger = logging.getLogger(__name__)


class ClaraRAG(RAGBackend):
    """RAG using Apple CLaRa for compression-native retrieval + generation.

    Uses CLaRa-7B-E2E for end-to-end retrieval and answer generation.
    Falls back to in-memory storage when model is unavailable.
    """

    def __init__(
        self,
        model_name: str = "apple/CLaRa-7B-E2E",
        device: Optional[str] = None,
        generation_top_k: int = 5,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.generation_top_k = generation_top_k
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._documents: List[List[str]] = []
        self._load_model(device)

    def _load_model(self, device: Optional[str]) -> None:
        try:
            import torch
            from transformers import AutoModel

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLaRa model: {self.model_name}")
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(device)
            self._model.eval()
            self._device = device
            logger.info("CLaRa model loaded")
        except Exception as e:
            logger.warning(f"Failed to load CLaRa: {e}. Using fallback.")
            self._model = None
            self._documents = []

    def index(self, documents: List[str], document_ids: List[str] | None = None) -> None:
        # CLaRa expects lists of document strings per query; we store as candidate pool
        if self._model is None:
            self._documents = [[d] for d in documents]
            return
        # For CLaRa E2E, we store raw docs; retrieval happens at query time via model
        self._documents = self._chunk_for_clara(documents)

    def _chunk_for_clara(self, docs: List[str], max_len: int = 512) -> List[List[str]]:
        out = []
        for d in docs:
            words = d.split()
            chunk = []
            for w in words:
                chunk.append(w)
                if len(" ".join(chunk)) >= max_len:
                    out.append([" ".join(chunk)])
                    chunk = []
            if chunk:
                out.append([" ".join(chunk)])
        return out if out else [[""]]

    def query(self, question: str, top_k: int = 5) -> tuple[str, List[str]]:
        if self._model is None:
            return (
                "CLaRa model not available. Please install and load CLaRa-7B-E2E.",
                [],
            )
        try:
            import torch

            # Build candidate docs: use top_k from our stored docs
            candidates = []
            for doc_list in self._documents[: max(top_k, 20)]:
                candidates.extend(doc_list)
            if not candidates:
                return "No documents indexed.", []

            questions = [question]
            documents = [candidates[:20]]  # CLaRa expects list of doc lists
            with torch.no_grad():
                output, topk_indices = self._model.generate_from_questions(
                    questions=questions,
                    documents=documents,
                    max_new_tokens=self.max_new_tokens,
                )
            answer = output[0] if output else ""
            retrieved = [candidates[i] for i in (topk_indices[0] if topk_indices else [])]
            return answer, retrieved
        except Exception as e:
            logger.error(f"CLaRa query failed: {e}")
            return f"Query failed: {e}", []
