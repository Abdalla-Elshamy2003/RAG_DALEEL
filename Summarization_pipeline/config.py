from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# project root = parent of Summarization_pipeline
# This ensures that load_dotenv looks for the .env file in the correct folder.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    # ── Database ───────────────────────────────────────────────────────────────
    # SENIOR TIP: For production, move this string into your .env file!
    db_conn: str = field(
        default_factory=lambda: os.environ.get(
            "DB_CONN",
            "host=135.181.117.17 port=5432 dbname=docs_ingestion user=abdu password=Neurix@123!@#"
        )
    )

    # ── Local Ollama  ───────────────────────────────────────────
    # Run 'ollama pull qwen2.5:7b' in your terminal to download the model first.
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_URL", 
            "http://localhost:11434"
        )
    )

    ollama_model: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_MODEL", 
            "qcwind/qwen2.5-7B-instruct-Q4_K_M"
        )
    )

    # ── Embedding (Downloaded Model) ───────────────────────────────────────────
    # BAAI/bge-m3 will automatically download to ~/.cache/huggingface on first run
    embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBED_MODEL",
            "BAAI/bge-m3"
        )
    )

    # CRITICAL: Kept on CPU to leave the GPU's 6GB VRAM entirely for Ollama
    embed_device: str = field(
        default_factory=lambda: os.environ.get("EMBED_DEVICE", "cpu")
    )

    embed_batch_size: int = 4  # Small batch size for stable CPU execution

    # ── LLM Generation Settings ────────────────────────────────────────────────
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1

    # ── Pipeline Behaviour ─────────────────────────────────────────────────────
    skip_done: bool = True
    cluster_similarity_threshold: float = 0.75

    # ── Table names ────────────────────────────────────────────────────────────
    table_documents: str = "post_processing_data"
    table_parent_chunks: str = "parent_chunks"
    table_child_chunks: str = "child_chunks"
    table_summaries: str = "summaries"
    table_pipeline_runs: str = "summarization_pipeline_runs"

    # ── Parent chunk columns ───────────────────────────────────────────────────
    col_text: str = "text"
    col_position: str = "parent_index"


# Instantiate the singleton config object
config = AppConfig()