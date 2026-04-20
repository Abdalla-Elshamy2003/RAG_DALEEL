"""
Unified Pipeline: Preprocessing → Chunking → Embedding
──────────────────────────────────────────────────────
Single script that runs the complete document processing pipeline:
1. Preprocessing: Extract text from PDF/DOCX/TXT files
2. Chunking: Split documents into parent/child chunks
3. Embedding: Generate vector embeddings for semantic search

Usage:
    python run_pipeline.py --folder ./data --limit 100

Environment variables:
    DB_CONN          PostgreSQL connection string
    WATCH_FOLDER     Default folder to process (default: ./data)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg

from chunking import ChunkConfig, run_chunking
from embedding import EmbeddingConfig, run_incremental_embeddings
from embedding.config import EMBEDDING_MODEL_LABEL, EMBEDDING_MODEL_NAME
from ingest_app.config import AppConfig
from ingest_app.db import create_tables, insert_payload
from preprocessing import run_ingestion
from preprocessing.file_utils import iter_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("unified_pipeline")


def run_full_pipeline(
    folder: str = "./data",
    limit: int | None = None,
    skip_preprocessing: bool = False,
    skip_chunking: bool = False,
    skip_embedding: bool = False,
    chunk_config: ChunkConfig | None = None,
    embed_config: EmbeddingConfig | None = None,
) -> int:
    """
    Run the complete pipeline: preprocessing → chunking → embedding.

    Args:
        folder: Path to folder containing documents
        limit: Maximum documents to process (None = all)
        skip_preprocessing: Skip preprocessing step
        skip_chunking: Skip chunking step
        skip_embedding: Skip embedding step
        chunk_config: Configuration for chunking
        embed_config: Configuration for embedding

    Returns:
        Exit code (0 = success, 1+ = errors)
    """
    cfg = AppConfig()
    chunk_cfg = chunk_config or ChunkConfig()
    embed_cfg = embed_config or EmbeddingConfig()

    # ─────────────────────────────────────────────
    # STEP 1: Preprocessing
    # ─────────────────────────────────────────────
    if not skip_preprocessing:
        log.info("=" * 60)
        log.info("STEP 1: PREPROCESSING")
        log.info("=" * 60)

        root = Path(folder)
        if not root.exists():
            log.error(f"Folder not found: {root}")
            return 1

        files = list(iter_files(root))
        if limit:
            files = files[:limit]

        log.info(f"Found {len(files)} files to process")

        # Use run_ingestion which handles DB insertion
        result = run_ingestion(folder, cfg, limit=limit)
        if result != 0:
            log.error(f"Preprocessing failed with exit code {result}")
            return result

        log.info("✅ Preprocessing completed")
    else:
        log.info("⏭️  Skipping preprocessing")

    # ─────────────────────────────────────────────
    # STEP 2: Chunking
    # ─────────────────────────────────────────────
    if not skip_chunking:
        log.info("=" * 60)
        log.info("STEP 2: CHUNKING")
        log.info("=" * 60)

        result = run_chunking(
            db_conn=cfg.db_conn,
            tokenizer_model=embed_cfg.model_name,
            limit=limit,
            config=chunk_cfg,
            log_level="INFO",
        )
        if result != 0:
            log.error(f"Chunking failed with exit code {result}")
            return result

        log.info("✅ Chunking completed")
    else:
        log.info("⏭️  Skipping chunking")

    # ─────────────────────────────────────────────
    # STEP 3: Embedding
    # ─────────────────────────────────────────────
    if not skip_embedding:
        log.info("=" * 60)
        log.info("STEP 3: EMBEDDING")
        log.info("=" * 60)

        result = run_incremental_embeddings(
            db_conn=cfg.db_conn,
            model_name=embed_cfg.model_name,
            config=embed_cfg,
        )
        if result != 0:
            log.error(f"Embedding failed with exit code {result}")
            return result

        log.info("✅ Embedding completed")
    else:
        log.info("⏭️  Skipping embedding")

    log.info("=" * 60)
    log.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
    log.info("=" * 60)

    return 0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline: Preprocessing -> Chunking -> Embedding"
    )
    parser.add_argument(
        "--folder",
        default="./data",
        help="Folder containing documents to process (default: ./data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking step",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding step",
    )

    # Chunking options
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        default=1800,
        help="Parent chunk size in tokens (default: 1800)",
    )
    parser.add_argument(
        "--parent-chunk-overlap",
        type=int,
        default=300,
        help="Parent chunk overlap in tokens (default: 300)",
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=600,
        help="Child chunk size in tokens (default: 600)",
    )
    parser.add_argument(
        "--child-chunk-overlap",
        type=int,
        default=120,
        help="Child chunk overlap in tokens (default: 120)",
    )

    # Embedding options
    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL_NAME,
        help="Embedding model path or Hugging Face id",
    )
    parser.add_argument(
        "--model-label",
        default=EMBEDDING_MODEL_LABEL,
        help="Canonical embedding model label stored in the database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size (default: 8)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    chunk_cfg = ChunkConfig(
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
    )

    embed_cfg = EmbeddingConfig(
        model_name=args.model_name,
        model_label=args.model_label,
        batch_size=args.batch_size,
    )

    exit_code = run_full_pipeline(
        folder=args.folder,
        limit=args.limit,
        skip_preprocessing=args.skip_preprocessing,
        skip_chunking=args.skip_chunking,
        skip_embedding=args.skip_embedding,
        chunk_config=chunk_cfg,
        embed_config=embed_cfg,
    )

    sys.exit(exit_code)
