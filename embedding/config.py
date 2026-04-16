from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_SOURCE_MODEL = "BAAI/bge-m3"
DEFAULT_LOCAL_MODEL_DIR = PROJECT_ROOT / "local_models" / "BAAI__bge-m3"
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    str(DEFAULT_LOCAL_MODEL_DIR),
)
EMBEDDING_MODEL_LABEL = os.environ.get(
    "EMBEDDING_MODEL_LABEL",
    EMBEDDING_SOURCE_MODEL,
)
EMBEDDING_VERSION = 2


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = EMBEDDING_MODEL_NAME
    model_label: str = EMBEDDING_MODEL_LABEL
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
