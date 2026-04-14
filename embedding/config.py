from __future__ import annotations

from dataclasses import dataclass

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_VERSION = 1


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = EMBEDDING_MODEL_NAME
    embedding_version: int = EMBEDDING_VERSION
    batch_size: int = 8
    fetch_limit: int = 128
    max_length: int = 8192
    use_fp16: bool = False
    normalize_vectors: bool = False
    create_hnsw_indexes: bool = True
    encode_parent_chunks: bool = True
    encode_child_chunks: bool = True
    log_level: str = "INFO"