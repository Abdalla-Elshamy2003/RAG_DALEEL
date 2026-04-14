from __future__ import annotations

import argparse

from . import EmbeddingConfig, run_incremental_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run embeddings only")
    parser.add_argument("--db-conn", default=None)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fetch-limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)

    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--normalize-vectors", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--skip-parent", action="store_true")
    parser.add_argument("--skip-child", action="store_true")

    args = parser.parse_args()

    cfg = EmbeddingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        fetch_limit=args.fetch_limit,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        normalize_vectors=args.normalize_vectors,
        create_hnsw_indexes=not args.no_index,
        encode_parent_chunks=not args.skip_parent,
        encode_child_chunks=not args.skip_child,
        log_level=args.log_level,
    )

    return run_incremental_embeddings(
        db_conn=args.db_conn,
        model_name=args.model_name,
        only_doc_id=args.doc_id,
        config=cfg,
    )


if __name__ == "__main__":
    raise SystemExit(main())