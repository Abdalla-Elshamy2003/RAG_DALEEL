from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

from config import config

logger = logging.getLogger(__name__)

_pool: Optional[pool.ThreadedConnectionPool] = None


def init_pool(min_conn: int = 2, max_conn: int = 8) -> None:
    global _pool
    # إخفاء كلمة المرور في السجلات لأغراض أمنية
    safe_dsn = config.db_conn.split("password=")[0].strip()
    _pool = pool.ThreadedConnectionPool(min_conn, max_conn, config.db_conn)
    logger.info("DB pool ready — %s", safe_dsn)


def _get_pool() -> pool.ThreadedConnectionPool:
    if _pool is None:
        raise RuntimeError("Call db.init_pool() before any DB operations.")
    return _pool


@contextmanager
def get_conn():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = True):
    factory = RealDictCursor if dict_cursor else None
    with get_conn() as conn:
        with conn.cursor(cursor_factory=factory) as cur:
            yield cur


# ── Documents ─────────────────────────────────────────────────────────────────

def fetch_all_doc_ids() -> list[int]:
    with get_cursor() as cur:
        cur.execute(f"SELECT id FROM {config.table_documents} ORDER BY id")
        return [r["id"] for r in cur.fetchall()]


def fetch_doc_metadata(doc_pk: int) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT id, doc_id, file_name, file_path, file_ext, file_hash,
                   source_type, extraction_status, language, page_count, payload, created_at
            FROM {config.table_documents}
            WHERE id = %s
            """,
            (doc_pk,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ── Parent chunks ─────────────────────────────────────────────────────────────

def fetch_parents_for_doc(doc_id: str) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT id, parent_id, doc_id, parent_index, {config.col_text} AS text,
                   token_count, char_count, start_char, end_char, metadata,
                   created_at, updated_at, COALESCE(metadata->>'language', NULL) AS language
            FROM {config.table_parent_chunks}
            WHERE doc_id = %s
            ORDER BY {config.col_position}
            """,
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ── Summaries ─────────────────────────────────────────────────────────────────

def already_summarized(level: int, source_id: Optional[int]) -> bool:
    with get_cursor() as cur:
        if source_id is None: return False
        cur.execute(
            f"SELECT 1 FROM {config.table_summaries} WHERE level = %s AND source_id = %s AND status = 'done' LIMIT 1",
            (level, source_id),
        )
        return cur.fetchone() is not None


def upsert_summary(
    level: int,
    source_id: Optional[int],
    summary_text: str,
    metadata: dict,
    embedding: Optional[list[float]] = None,
    cluster_doc_ids: Optional[list[int]] = None,
) -> int:
    with get_cursor() as cur:
        # للمستوى الثالث: نقوم بالإدخال فقط لأن source_id يكون فارغاً
        if level == 3:
            cur.execute(
                f"""
                INSERT INTO {config.table_summaries}
                    (level, source_id, summary_text, metadata, embedding, cluster_doc_ids, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'done')
                RETURNING id
                """,
                (level, source_id, summary_text, Json(metadata), embedding, cluster_doc_ids),
            )
            return cur.fetchone()["id"]

        # للمستويات 1 و 2: نستخدم ON CONFLICT لتحديث البيانات إذا كانت موجودة (Senior Move)
        cur.execute(
            f"""
            INSERT INTO {config.table_summaries}
                (level, source_id, summary_text, metadata, embedding, cluster_doc_ids, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'done')
            ON CONFLICT (level, source_id) WHERE level < 3
            DO UPDATE SET
                summary_text    = EXCLUDED.summary_text,
                metadata        = EXCLUDED.metadata,
                embedding       = EXCLUDED.embedding,
                cluster_doc_ids = EXCLUDED.cluster_doc_ids,
                status          = 'done',
                error_message   = NULL,
                updated_at      = NOW()
            RETURNING id
            """,
            (level, source_id, summary_text, Json(metadata), embedding, cluster_doc_ids),
        )
        return cur.fetchone()["id"]


def fetch_level1_summaries_for_doc(doc_pk: int) -> list[dict]:
    """
    مهمة جداً: تجلب ملخصات الأجزاء لبناء ملخص المستند (Level 2).
    """
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT id, source_id, summary_text, metadata
            FROM {config.table_summaries}
            WHERE level = 1
              AND (metadata->>'doc_pk')::BIGINT = %s
              AND status = 'done'
            """,
            (doc_pk,),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_l2_summaries_with_embeddings() -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT source_id as doc_id, summary_text, embedding, metadata
            FROM {config.table_summaries}
            WHERE level = 2 
              AND status = 'done' 
              AND embedding IS NOT NULL
            ORDER BY id
            """
        )
        return [dict(r) for r in cur.fetchall()]


def delete_summaries_at_level(level: int) -> None:
    with get_cursor() as cur:
        cur.execute(f"DELETE FROM {config.table_summaries} WHERE level = %s", (level,))
        logger.info(f"Deleted old Level {level} summaries.")


# ── Pipeline runs ─────────────────────────────────────────────────────────────

def start_pipeline_run(run_type: str) -> int:
    with get_cursor() as cur:
        cur.execute(
            f"INSERT INTO {config.table_pipeline_runs} (run_type) VALUES (%s) RETURNING id",
            (run_type,),
        )
        return cur.fetchone()["id"]


def finish_pipeline_run(run_id: int, docs: int, summaries: int, error: Optional[str] = None) -> None:
    with get_cursor() as cur:
        cur.execute(
            f"""
            UPDATE {config.table_pipeline_runs}
            SET status            = %s,
                docs_processed    = %s,
                summaries_created = %s,
                error_message     = %s,
                finished_at       = NOW()
            WHERE id = %s
            """,
            ("failed" if error else "done", docs, summaries, error, run_id),
        )