from __future__ import annotations

import logging
import uuid

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore

try:
    from ingest_app.config import AppConfig
except ModuleNotFoundError:
    from config import AppConfig

from .chunking_db_langchain import (
    create_chunk_tables,
    delete_chunks_for_doc,
    fetch_documents_needing_chunking,
    get_chunk_stats,
    insert_child_chunks,
    insert_parent_chunks,
)
from .config import ChunkConfig, STRATEGY_NAME, STRATEGY_VERSION
from .recursive_chunker import RecursiveParentChildChunker


def _resolve_db_conn(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return AppConfig().db_conn


def run_chunking(
    *,
    db_conn: str | None = None,
    tokenizer_model: str = "BAAI/bge-m3",
    limit: int | None = None,
    doc_id: str | None = None,
    file_hash: str | None = None,
    config: ChunkConfig | None = None,
    log_level: str = "INFO",
) -> int:
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install: pip install 'psycopg[binary]'")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("langchain_chunker")

    resolved_db_conn = _resolve_db_conn(db_conn)
    cfg = config or ChunkConfig()
    build_run_id = f"chunk_run_{uuid.uuid4().hex[:12]}"

    chunker = RecursiveParentChildChunker(tokenizer_model=tokenizer_model, config=cfg)

    with psycopg.connect(resolved_db_conn) as conn:
        create_chunk_tables(conn)

        docs = fetch_documents_needing_chunking(
            conn,
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            tokenizer_model=tokenizer_model,
            parent_chunk_size=cfg.parent_chunk_size,
            parent_chunk_overlap=cfg.parent_chunk_overlap,
            child_chunk_size=cfg.child_chunk_size,
            child_chunk_overlap=cfg.child_chunk_overlap,
            min_child_chunk_tokens=cfg.min_child_chunk_tokens,
            limit=limit,
            only_doc_id=doc_id,
            only_hash=file_hash,
        )

        if not docs:
            log.info("No documents need chunking.")
            return 0

        log.info("build_run_id=%s documents_to_chunk=%d", build_run_id, len(docs))

        processed = 0
        failed = 0

        for doc in docs:
            doc_id_value = str(doc["doc_id"])
            try:
                parents, children = chunker.build_rows_for_document(doc, build_run_id=build_run_id)

                delete_chunks_for_doc(conn, doc_id_value)
                insert_parent_chunks(conn, parents)
                insert_child_chunks(conn, children)
                conn.commit()

                processed += 1
                log.info(
                    "Chunked doc_id=%s file=%s parents=%d children=%d",
                    doc_id_value,
                    doc.get("file_name"),
                    len(parents),
                    len(children),
                )
            except Exception:
                conn.rollback()
                failed += 1
                log.exception("Failed chunking doc_id=%s file=%s", doc_id_value, doc.get("file_name"))

        stats = get_chunk_stats(conn)
        log.info(
            "processed=%d failed=%d parent_total=%d child_total=%d",
            processed,
            failed,
            stats["parent_total"],
            stats["child_total"],
        )

    return 0 if failed == 0 else 3