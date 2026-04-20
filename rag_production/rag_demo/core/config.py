from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH, override=True)


def _reload_env() -> None:
    load_dotenv(_ENV_PATH, override=True)


def _get_str(name: str, default: str) -> str:
    _reload_env()
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    _reload_env()
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    _reload_env()
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _get_float_or_none(name: str) -> Optional[float]:
    _reload_env()
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _get_bool(name: str, default: bool) -> bool:
    _reload_env()
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RAGConfig:
    # Database
    db_conn: str = field(default_factory=lambda: _get_str("DB_CONN", ""))

    # Embedding model
    embedding_model: str = field(
        default_factory=lambda: _get_str("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    )
    embedding_version: int = field(
        default_factory=lambda: _get_int("EMBEDDING_VERSION", 1)
    )

    # Reranker model
    reranker_model: str = field(
        default_factory=lambda: _get_str(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-v2-m3",
        )
    )
    reranker_max_chars: int = field(
        default_factory=lambda: _get_int("RERANKER_MAX_CHARS", 2500)
    )
    rerank_top_k: int = field(default_factory=lambda: _get_int("RERANK_TOP_K", 3))
    cpu_skip_reranker: bool = field(
        default_factory=lambda: _get_bool("CPU_SKIP_RERANKER", True)
    )

    # Hybrid retrieval
    vector_candidates: int = field(
        default_factory=lambda: _get_int("VECTOR_CANDIDATES", 24)
    )
    keyword_candidates: int = field(
        default_factory=lambda: _get_int("KEYWORD_CANDIDATES", 24)
    )
    fused_child_limit: int = field(
        default_factory=lambda: _get_int("FUSED_CHILD_LIMIT", 12)
    )
    parent_limit: int = field(default_factory=lambda: _get_int("PARENT_LIMIT", 6))
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))

    # pgvector HNSW
    hnsw_ef_search: int = field(
        default_factory=lambda: _get_int("HNSW_EF_SEARCH", 100)
    )

    # Web fallback
    web_fallback_enabled: bool = field(
        default_factory=lambda: _get_bool("WEB_FALLBACK_ENABLED", False)
    )
    tavily_api_key: str = field(default_factory=lambda: _get_str("TAVILY_API_KEY", ""))
    tavily_max_results: int = field(
        default_factory=lambda: _get_int("TAVILY_MAX_RESULTS", 3)
    )
    tavily_search_depth: str = field(
        default_factory=lambda: _get_str("TAVILY_SEARCH_DEPTH", "basic")
    )
    web_timeout_seconds: float = field(
        default_factory=lambda: _get_float("WEB_TIMEOUT_SECONDS", 5.0)
    )

    # Confidence threshold
    # Keep None until you evaluate reranker scores.
    internal_min_rerank_score: Optional[float] = field(
        default_factory=lambda: _get_float_or_none("INTERNAL_MIN_RERANK_SCORE")
    )

    # Open-source local LLM using Ollama
    llm_provider: str = field(
        default_factory=lambda: _get_str("LLM_PROVIDER", "ollama").lower()
    )
    ollama_base_url: str = field(
        default_factory=lambda: _get_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _get_str("OLLAMA_MODEL", "qwen2.5:3b")
    )
    ollama_connect_timeout_seconds: float = field(
        default_factory=lambda: _get_float("OLLAMA_CONNECT_TIMEOUT_SECONDS", 10.0)
    )
    ollama_request_timeout_seconds: float = field(
        default_factory=lambda: _get_float("OLLAMA_REQUEST_TIMEOUT_SECONDS", 600.0)
    )
    ollama_stream_timeout_seconds: float = field(
        default_factory=lambda: _get_float("OLLAMA_STREAM_TIMEOUT_SECONDS", 600.0)
    )
    ollama_num_ctx: int = field(default_factory=lambda: _get_int("OLLAMA_NUM_CTX", 4096))
    ollama_num_predict: int = field(
        default_factory=lambda: _get_int("OLLAMA_NUM_PREDICT", 128)
    )
    ollama_keep_alive: str = field(
        default_factory=lambda: _get_str("OLLAMA_KEEP_ALIVE", "30m")
    )
    prompt_context_limit: int = field(
        default_factory=lambda: _get_int("PROMPT_CONTEXT_LIMIT", 2)
    )
    allow_general_knowledge_fallback: bool = field(
        default_factory=lambda: _get_bool("ALLOW_GENERAL_KNOWLEDGE_FALLBACK", True)
    )

    # Answer behavior
    answer_language: str = field(
        default_factory=lambda: _get_str("ANSWER_LANGUAGE", "same_as_question")
    )
    default_answer_style: str = field(
        default_factory=lambda: _get_str(
            "ANSWER_STYLE",
            "detailed, structured, grounded, and explicit about evidence",
        )
    )

    def validate(self) -> None:
        if not self.db_conn:
            raise ValueError("DB_CONN is missing. Check your .env file.")

        if self.llm_provider != "ollama":
            raise ValueError(
                "This production setup is configured for open-source local models. "
                "Use LLM_PROVIDER=ollama in .env."
            )

        if self.ollama_connect_timeout_seconds <= 0:
            raise ValueError("OLLAMA_CONNECT_TIMEOUT_SECONDS must be greater than 0.")

        if self.ollama_request_timeout_seconds <= 0:
            raise ValueError("OLLAMA_REQUEST_TIMEOUT_SECONDS must be greater than 0.")

        if self.ollama_stream_timeout_seconds <= 0:
            raise ValueError("OLLAMA_STREAM_TIMEOUT_SECONDS must be greater than 0.")

        if self.ollama_num_ctx <= 0:
            raise ValueError("OLLAMA_NUM_CTX must be greater than 0.")

        if self.web_timeout_seconds <= 0:
            raise ValueError("WEB_TIMEOUT_SECONDS must be greater than 0.")

        if self.ollama_num_predict <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT must be greater than 0.")

        if self.prompt_context_limit <= 0:
            raise ValueError("PROMPT_CONTEXT_LIMIT must be greater than 0.")
