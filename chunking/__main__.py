from __future__ import annotations

import argparse

from . import ChunkConfig, run_chunking
from embedding.config import EMBEDDING_MODEL_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chunking only")
    parser.add_argument("--db-conn", default=None)
    parser.add_argument("--tokenizer-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--file-hash", default=None)
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--parent-chunk-size", type=int, default=1800)
    parser.add_argument("--parent-chunk-overlap", type=int, default=300)
    parser.add_argument("--child-chunk-size", type=int, default=600)
    parser.add_argument("--child-chunk-overlap", type=int, default=120)
    parser.add_argument("--min-child-chunk-tokens", type=int, default=200)

    args = parser.parse_args()

    cfg = ChunkConfig(
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        min_child_chunk_tokens=args.min_child_chunk_tokens,
    )

    return run_chunking(
        db_conn=args.db_conn,
        tokenizer_model=args.tokenizer_model,
        limit=args.limit,
        doc_id=args.doc_id,
        file_hash=args.file_hash,
        config=cfg,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
