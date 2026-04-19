from __future__ import annotations

import logging
from typing import List

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import RAGConfig
from .schemas import RetrievedContext

log = logging.getLogger(__name__)


class GPUModelManager:
    """
    Loads and manages open-source embedding + reranker models.

    Embedding model:
        BAAI/bge-m3

    Reranker model:
        BAAI/bge-reranker-v2-m3
    """

    _instance = None

    def __new__(cls, config: RAGConfig):
        if cls._instance is None:
            cls._instance = super(GPUModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: RAGConfig):
        if self._initialized:
            return

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Using device: %s", self.device)

        log.info("Loading embedding model: %s", config.embedding_model)
        self.encoder = SentenceTransformer(
            config.embedding_model,
            device=self.device,
        )

        log.info("Loading reranker model: %s", config.reranker_model)
        self.reranker = CrossEncoder(
            config.reranker_model,
            device=self.device,
            max_length=512,
        )

        if self.device == "cuda":
            try:
                self.encoder.half()
                self.reranker.model.half()
                log.info("Models converted to FP16.")
            except Exception:
                log.warning("Could not convert models to FP16. Continuing with default precision.")

        self._initialized = True

    def encode_query(self, query: str) -> List[float]:
        """
        Encode user query into a normalized 1024-dim vector for BAAI/bge-m3.
        """
        query = (query or "").strip()

        if not query:
            raise ValueError("Query is empty.")

        vector = self.encoder.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vector.tolist()

    def rerank(
        self,
        *,
        query: str,
        contexts: List[RetrievedContext],
        top_k: int | None = None,
    ) -> List[RetrievedContext]:
        """
        Rerank retrieved contexts using cross-encoder reranker.

        Input:
            query + candidate context

        Output:
            same contexts sorted by rerank_score descending.
        """
        query = (query or "").strip()

        if not query:
            raise ValueError("Query is empty.")

        if not contexts:
            return []

        top_k = top_k or self.config.rerank_top_k

        pairs = [
            [
                query,
                context.to_rerank_text(
                    max_chars=self.config.reranker_max_chars
                ),
            ]
            for context in contexts
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        for context, score in zip(contexts, scores):
            context.rerank_score = float(score)

        contexts.sort(
            key=lambda item: (
                item.rerank_score
                if item.rerank_score is not None
                else -999999.0
            ),
            reverse=True,
        )

        return contexts[:top_k]