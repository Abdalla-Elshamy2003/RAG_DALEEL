from __future__ import annotations

import hashlib

from .config import EmbeddingConfig
from .db import EMBEDDING_DIM
from .utils import _l2_normalize


class BGEEmbeddingModel:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._backend = "flagembedding"
        self._use_hash_backend = config.model_name.startswith("local-hash")

        if self._use_hash_backend:
            self._backend = "local-hash"
            self.model = None
            return

        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
            self.model = BGEM3FlagModel(
                config.model_name,
                use_fp16=config.use_fp16,
            )
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Embedding backend missing. Install one of: "
                    "pip install FlagEmbedding OR pip install sentence-transformers"
                ) from exc

            self._backend = "sentence-transformers"
            self.model = SentenceTransformer(config.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._backend == "flagembedding":
            output = self.model.encode(
                texts,
                batch_size=len(texts),
                max_length=self.config.max_length,
            )
            dense_vectors = output["dense_vecs"]
            vectors = [[float(x) for x in vec] for vec in dense_vectors]
        elif self._backend == "local-hash":
            vectors = [self._hash_embedding(text) for text in texts]
        else:
            dense_vectors = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            vectors = [[float(x) for x in vec] for vec in dense_vectors]

        vectors = [self._fit_dim(vec) for vec in vectors]

        if self.config.normalize_vectors:
            vectors = [_l2_normalize(vec) for vec in vectors]

        return vectors

    @staticmethod
    def _fit_dim(vec: list[float]) -> list[float]:
        if len(vec) == EMBEDDING_DIM:
            return vec
        if len(vec) > EMBEDDING_DIM:
            return vec[:EMBEDDING_DIM]
        return vec + [0.0] * (EMBEDDING_DIM - len(vec))

    @staticmethod
    def _hash_embedding(text: str) -> list[float]:
        if not text:
            return [0.0] * EMBEDDING_DIM

        values: list[float] = []
        seed = text.encode("utf-8", errors="ignore")
        counter = 0
        while len(values) < EMBEDDING_DIM:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for i in range(0, len(digest), 2):
                n = int.from_bytes(digest[i:i + 2], "little", signed=False)
                values.append((n / 65535.0) * 2.0 - 1.0)
                if len(values) >= EMBEDDING_DIM:
                    break
            counter += 1
        return values