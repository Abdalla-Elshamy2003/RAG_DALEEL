from __future__ import annotations

from dataclasses import dataclass

STRATEGY_NAME = "langchain_recursive_token_overlap_parent_child"
STRATEGY_VERSION = 4
SEMANTIC_MODEL = "none"
DEFAULT_SEPARATORS = ["\\n\\n", "\\n", ". ", "؟ ", "! ", "؛ ", "، ", " ", ""]


@dataclass(slots=True)
class ChunkConfig:
    parent_chunk_size: int = 1800
    parent_chunk_overlap: int = 300
    child_chunk_size: int = 600
    child_chunk_overlap: int = 120
    min_child_chunk_tokens: int = 200
    separators: tuple[str, ...] = tuple(DEFAULT_SEPARATORS)