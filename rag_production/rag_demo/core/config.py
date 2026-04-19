from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv(".env", override=True)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _get_float_or_none(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RAGConfig:
    # Database
    db_conn: str = os.getenv("DB_CONN", "")

    # Embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_version: int = _get_int("EMBEDDING_VERSION", 1)

    # Reranker model
    reranker_model: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    reranker_max_chars: int = _get_int("RERANKER_MAX_CHARS", 6000)
    rerank_top_k: int = _get_int("RERANK_TOP_K", 5)

    # Hybrid retrieval
    vector_candidates: int = _get_int("VECTOR_CANDIDATES", 80)
    keyword_candidates: int = _get_int("KEYWORD_CANDIDATES", 80)
    fused_child_limit: int = _get_int("FUSED_CHILD_LIMIT", 40)
    parent_limit: int = _get_int("PARENT_LIMIT", 10)
    rrf_k: int = _get_int("RRF_K", 60)

    # pgvector HNSW
    hnsw_ef_search: int = _get_int("HNSW_EF_SEARCH", 100)

    # Web fallback
    web_fallback_enabled: bool = _get_bool("WEB_FALLBACK_ENABLED", True)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    tavily_max_results: int = _get_int("TAVILY_MAX_RESULTS", 5)

    # Confidence threshold
    # Keep None until you evaluate reranker scores.
    internal_min_rerank_score: Optional[float] = _get_float_or_none(
        "INTERNAL_MIN_RERANK_SCORE"
    )

    # Open-source local LLM using Ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    # Answer behavior
    answer_language: str = os.getenv("ANSWER_LANGUAGE", "same_as_question")
    default_answer_style: str = os.getenv(
        "ANSWER_STYLE",
        "professional, clear, grounded, and concise",
    )

    def validate(self) -> None:
        if not self.db_conn:
            raise ValueError("DB_CONN is missing. Check your .env file.")

        if self.llm_provider != "ollama":
            raise ValueError(
                "This production setup is configured for open-source local models. "
                "Use LLM_PROVIDER=ollama in .env."
            )