from __future__ import annotations

import argparse
import logging
import time

from . import EmbeddingConfig, run_incremental_embeddings
from .config import EMBEDDING_MODEL_LABEL, EMBEDDING_MODEL_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Run embeddings only")
    parser.add_argument("--db-conn", default=None)
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--model-label", default=EMBEDDING_MODEL_LABEL)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--watch", action="store_true", help="Run continuously, checking for new chunks every --interval seconds")
    parser.add_argument("--interval", type=int, default=30, help="Interval in seconds to check for new chunks when watching")

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
        model_label=args.model_label,
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

    if args.watch:
        logger = logging.getLogger("embedding_watcher")
        logger.info("Starting embedding watcher service, checking every %d seconds", args.interval)
        
        while True:
            try:
                result = run_incremental_embeddings(
                    db_conn=args.db_conn,
                    model_name=args.model_name,
                    only_doc_id=args.doc_id,
                    config=cfg,
                )
                if result == 0:
                    logger.info("No new chunks to embed, sleeping for %d seconds", args.interval)
                else:
                    logger.info("Processed embeddings, checking again in %d seconds", args.interval)
            except Exception as e:
                logger.error("Error in embedding watcher: %s", e)
            
            time.sleep(args.interval)
    else:
        return run_incremental_embeddings(
            db_conn=args.db_conn,
            model_name=args.model_name,
            only_doc_id=args.doc_id,
            config=cfg,
        )


if __name__ == "__main__":
    raise SystemExit(main())
