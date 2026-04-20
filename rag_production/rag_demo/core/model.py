# from __future__ import annotations

# import logging
# from typing import List

# import torch
# from sentence_transformers import CrossEncoder, SentenceTransformer

# from .config import RAGConfig
# from .schemas import RetrievedContext

# log = logging.getLogger(__name__)

# class GPUModelManager:
#     """
#     Loads and manages open-source embedding + reranker models.

#     Embedding model:
#         BAAI/bge-m3

#     Reranker model:
#         BAAI/bge-reranker-v2-m3
#     """

#     _instance = None

#     def __new__(cls, config: RAGConfig):
#         if cls._instance is None:
#             cls._instance = super(GPUModelManager, cls).__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def __init__(self, config: RAGConfig):
#         if self._initialized:
#             return

#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         log.info("Using device: %s", self.device)

#         log.info("Loading embedding model: %s", config.embedding_model)
#         self.encoder = SentenceTransformer(
#             config.embedding_model,
#             device=self.device,
#         )

#         log.info("Loading reranker model: %s", config.reranker_model)
#         self.reranker = CrossEncoder(
#             config.reranker_model,
#             device=self.device,
#             max_length=512,
#         )

#         if self.device == "cuda":
#             try:
#                 self.encoder.half()
#                 self.reranker.model.half()
#                 log.info("Models converted to FP16.")
#             except Exception:
#                 log.warning("Could not convert models to FP16. Continuing with default precision.")

#         self._initialized = True

#     def encode_query(self, query: str) -> List[float]:
#         """
#         Encode user query into a normalized 1024-dim vector for BAAI/bge-m3.
#         """
#         query = (query or "").strip()

#         if not query:
#             raise ValueError("Query is empty.")

#         vector = self.encoder.encode(
#             query,
#             convert_to_tensor=False,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         )

#         return vector.tolist()

#     def rerank(
#         self,
#         *,
#         query: str,
#         contexts: List[RetrievedContext],
#         top_k: int | None = None,
#     ) -> List[RetrievedContext]:
#         """
#         Rerank retrieved contexts using cross-encoder reranker.

#         Input:
#             query + candidate context (only parent contexts)

#         Output:
#             same contexts sorted by rerank_score descending.
#         """
#         query = (query or "").strip()

#         if not query:
#             raise ValueError("Query is empty.")

#         if not contexts:
#             return []

#         top_k = top_k or self.config.rerank_top_k

#         # Filter out child contexts, keep only parent contexts
#         parent_contexts = [context for context in contexts if (context.text or "").strip()]

#         if not parent_contexts:
#             return []

#         pairs = [
#             [
#                 query,
#                 context.to_rerank_text(
#                     max_chars=self.config.reranker_max_chars
#                 ),
#             ]
#             for context in parent_contexts  # Rerank only parent contexts
#         ]

#         scores = self.reranker.predict(
#             pairs,
#             batch_size=8,
#             show_progress_bar=False,
#         )

#         # Assign rerank score to parent contexts only
#         for context, score in zip(parent_contexts, scores):
#             context.rerank_score = float(score)

#         # Sort contexts based on rerank_score
#         parent_contexts.sort(
#             key=lambda item: (
#                 item.rerank_score
#                 if item.rerank_score is not None
#                 else -999999.0
#             ),
#             reverse=True,
#         )

#         return parent_contexts[:top_k]

from __future__ import annotations

import logging
from pathlib import Path
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
            self._resolve_model_source(config.embedding_model),
            device=self.device,
            local_files_only=True,
        )

        log.info("Loading reranker model: %s", config.reranker_model)
        self.reranker = CrossEncoder(
            self._resolve_model_source(config.reranker_model),
            device=self.device,
            max_length=512,
            local_files_only=True,
        )

        if self.device == "cuda":
            try:
                self.encoder.half()
                self.reranker.model.half()
                log.info("Models converted to FP16.")
            except Exception:
                log.warning("Could not convert models to FP16. Continuing with default precision.")

        self._initialized = True

    def _resolve_model_source(self, model_name: str) -> str:
        parts = model_name.split("/", 1)
        if len(parts) != 2:
            return model_name

        org, repo = parts
        cache_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{org}--{repo}"
        )
        refs_main = cache_root / "refs" / "main"

        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = cache_root / "snapshots" / revision
            if snapshot_dir.exists():
                log.info("Using cached model snapshot: %s", snapshot_dir)
                return str(snapshot_dir)

        snapshots_dir = cache_root / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
            if snapshot_dirs:
                latest_snapshot = sorted(snapshot_dirs)[-1]
                log.info("Using cached model snapshot: %s", latest_snapshot)
                return str(latest_snapshot)

        return model_name

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
            query + candidate context (only parent contexts)

        Output:
            same contexts sorted by rerank_score descending.
        """
        query = (query or "").strip()

        if not query:
            raise ValueError("Query is empty.")

        if not contexts:
            return []

        top_k = top_k or self.config.rerank_top_k

        if self.device == "cpu" and self.config.cpu_skip_reranker:
            # On CPU the cross-encoder is the main latency source. Fall back to
            # fusion-score ordering so the system stays interactive.
            fast_contexts = [context for context in contexts if (context.text or "").strip()]
            for context in fast_contexts:
                context.rerank_score = float(context.fusion_score)
            fast_contexts.sort(
                key=lambda item: (
                    item.rerank_score if item.rerank_score is not None else -999999.0
                ),
                reverse=True,
            )
            return fast_contexts[:top_k]

        # Filter out child contexts, keep only parent contexts
        parent_contexts = [context for context in contexts if (context.text or "").strip()]

        if not parent_contexts:
            return []

        pairs = [
            [
                query,
                context.to_rerank_text(
                    max_chars=self.config.reranker_max_chars
                ),
            ]
            for context in parent_contexts  # Rerank only parent contexts
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        # Assign rerank score to parent contexts only
        for context, score in zip(parent_contexts, scores):
            context.rerank_score = float(score)

        # Sort contexts based on rerank_score
        parent_contexts.sort(
            key=lambda item: (
                item.rerank_score
                if item.rerank_score is not None
                else -999999.0
            ),
            reverse=True,
        )

        return parent_contexts[:top_k]
