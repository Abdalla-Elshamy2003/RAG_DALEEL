from __future__ import annotations

import logging

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore

try:
    from ingest_app.config import AppConfig
except ModuleNotFoundError:
    from config import AppConfig

from .config import EMBEDDING_MODEL_NAME, EmbeddingConfig
from .db import (
    CHILD_TABLE,
    PARENT_TABLE,
    ensure_hnsw_indexes,
    ensure_vector_schema,
    fetch_rows_needing_embedding,
    get_embedding_stats,
    update_embeddings,
)
from .model import BGEEmbeddingModel
from .utils import _batched, _default_use_fp16

def _table_has_pending_rows(
    conn: psycopg.Connection,
    *,
    table: str,
    id_col: str,
    config: EmbeddingConfig,
    only_doc_id: str | None = None,
) -> bool:
    rows = fetch_rows_needing_embedding(
        conn,
        table=table,
        id_col=id_col,
        model_name=config.model_name,
        embedding_version=config.embedding_version,
        limit=1,
        only_doc_id=only_doc_id,
    )
    return bool(rows)


def _backfill_table(
    conn: psycopg.Connection,
    *,
    model: BGEEmbeddingModel,
    table: str,
    id_col: str,
    config: EmbeddingConfig,
    only_doc_id: str | None = None,
    logger: logging.Logger,
) -> int:
    total_updated = 0

    while True:
        pending = fetch_rows_needing_embedding(
            conn,
            table=table,
            id_col=id_col,
            model_name=config.model_name,
            embedding_version=config.embedding_version,
            limit=config.fetch_limit,
            only_doc_id=only_doc_id,
        )
        # Release read transaction before running potentially long model inference.
        conn.commit()

        if not pending:
            break

        for batch in _batched(pending, config.batch_size):
            texts = [row["text"] for row in batch]
            row_ids = [str(row["row_id"]) for row in batch]
            vectors = model.encode(texts)

            update_rows = [
                (
                    row_id,
                    vector,
                    config.model_name,
                    config.embedding_version,
                )
                for row_id, vector in zip(row_ids, vectors)
            ]

            update_embeddings(
                conn,
                table=table,
                id_col=id_col,
                rows=update_rows,
            )
            conn.commit()
            total_updated += len(update_rows)

        logger.info("%s updated=%d", table, total_updated)

    return total_updated


def run_incremental_embeddings(
    *,
    db_conn: str | None = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    only_doc_id: str | None = None,
    config: EmbeddingConfig | None = None,
) -> int:
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install: pip install 'psycopg[binary]'")

    cfg = config or EmbeddingConfig(model_name=model_name)
    if config is None:
        cfg.use_fp16 = _default_use_fp16()

    resolved_db_conn = db_conn or AppConfig().db_conn

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("embedding_backfill")

    logger.info(
        "Embedding backfill started model=%s version=%s use_fp16=%s",
        cfg.model_name,
        cfg.embedding_version,
        cfg.use_fp16,
    )

    with psycopg.connect(resolved_db_conn) as conn:
        ensure_vector_schema(conn)

        parent_pending = _table_has_pending_rows(
            conn,
            table=PARENT_TABLE,
            id_col="parent_id",
            config=cfg,
            only_doc_id=only_doc_id,
        ) if cfg.encode_parent_chunks else False

        child_pending = _table_has_pending_rows(
            conn,
            table=CHILD_TABLE,
            id_col="child_id",
            config=cfg,
            only_doc_id=only_doc_id,
        ) if cfg.encode_child_chunks else False

        if not parent_pending and not child_pending:
            logger.info("No chunks need embeddings.")
            if cfg.create_hnsw_indexes:
                ensure_hnsw_indexes(conn)
            return 0

        embedder = BGEEmbeddingModel(cfg)

        updated_parent = 0
        updated_child = 0

        if parent_pending:
            updated_parent = _backfill_table(
                conn,
                model=embedder,
                table=PARENT_TABLE,
                id_col="parent_id",
                config=cfg,
                only_doc_id=only_doc_id,
                logger=logger,
            )

        if child_pending:
            updated_child = _backfill_table(
                conn,
                model=embedder,
                table=CHILD_TABLE,
                id_col="child_id",
                config=cfg,
                only_doc_id=only_doc_id,
                logger=logger,
            )

        if cfg.create_hnsw_indexes:
            ensure_hnsw_indexes(conn)

        stats = get_embedding_stats(conn)
        logger.info(
            "Embedding backfill finished updated_parent=%d updated_child=%d "
            "parent_embedded=%d/%d child_embedded=%d/%d",
            updated_parent,
            updated_child,
            stats["parent_embedded"],
            stats["parent_total"],
            stats["child_embedded"],
            stats["child_total"],
        )

    return 0