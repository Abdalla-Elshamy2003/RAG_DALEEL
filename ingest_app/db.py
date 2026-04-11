from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

# ─────────────────────────────────────────────
# Table names
# ─────────────────────────────────────────────
MAIN_TABLE = "preprocessing_data"
POST_TABLE = "post_processing_data"

 
_COLUMNS = """
    id                BIGSERIAL PRIMARY KEY,
    doc_id            TEXT NOT NULL UNIQUE,
    file_name         TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    file_ext          TEXT,
    file_hash         TEXT UNIQUE,
    source_type       TEXT,
    extraction_status TEXT,
    language          TEXT,
    page_count        INT,
    payload           JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
"""


def _create_one_table(conn: psycopg.Connection, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({_COLUMNS});")
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_payload_gin
                ON {table} USING GIN (payload);
        """)
        cur.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_file_hash_unique
                ON {table}(file_hash);
        """)
        # Drop old markdown_text column if exists
        cur.execute(f"""
            ALTER TABLE {table} DROP COLUMN IF EXISTS markdown_text;
        """)
    conn.commit()


def create_tables(conn: psycopg.Connection) -> None:
    """Ensure both preprocessing_data and post_processing_data exist."""
    _create_one_table(conn, MAIN_TABLE)
    _create_one_table(conn, POST_TABLE)


# ─────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────
def get_existing_hashes(conn: psycopg.Connection, hashes: list[str]) -> set[str]:
    if not hashes:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT file_hash FROM {MAIN_TABLE} WHERE file_hash = ANY(%s)",
            (hashes,),
        )
        return {row[0] for row in cur.fetchall() if row and row[0]}


def get_hash_by_filepath(conn: psycopg.Connection, file_paths: list[str]) -> dict[str, str]:
    """
    Returns {file_path: file_hash} for files already in DB.
    Used to detect modified files (same path, new hash = file changed).
    """
    if not file_paths:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT file_path, file_hash FROM {MAIN_TABLE} WHERE file_path = ANY(%s)",
            (file_paths,),
        )
        return {row[0]: row[1] for row in cur.fetchall() if row[0] and row[1]}


# ─────────────────────────────────────────────
# Insert helpers
# ─────────────────────────────────────────────
def _insert_into(conn: psycopg.Connection, table: str, payload: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table} (
                doc_id, file_name, file_path, file_ext, file_hash,
                source_type, extraction_status, language, page_count, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_hash) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                file_name = EXCLUDED.file_name,
                file_path = EXCLUDED.file_path,
                file_ext = EXCLUDED.file_ext,
                source_type = EXCLUDED.source_type,
                extraction_status = EXCLUDED.extraction_status,
                language = EXCLUDED.language,
                page_count = EXCLUDED.page_count,
                payload = EXCLUDED.payload
        """, (
            payload.get("doc_id"),
            payload.get("file_name"),
            payload.get("file_path"),
            Path(payload.get("file_path", "")).suffix.lower() or None,
            payload.get("file_hash"),
            payload.get("source_type"),
            payload.get("extraction_status"),
            payload.get("language"),
            payload.get("page_count"),
            Jsonb(payload),
        ))


def insert_payload(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    _insert_into(conn, MAIN_TABLE, payload)


def insert_post_processing_payload(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    _insert_into(conn, POST_TABLE, payload)


# ─────────────────────────────────────────────
# Sync: preprocessing_data → post_processing_data
# ─────────────────────────────────────────────
def sync_post_processing_from_main(conn: psycopg.Connection) -> int:
    """
    Copy or update every row from preprocessing_data to post_processing_data.
    Returns the number of rows affected.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {POST_TABLE} (
                doc_id, file_name, file_path, file_ext, file_hash,
                source_type, extraction_status, language, page_count, payload
            )
            SELECT
                doc_id, file_name, file_path, file_ext, file_hash,
                source_type, extraction_status, language, page_count, payload
            FROM {MAIN_TABLE}
            ON CONFLICT (file_hash) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                file_name = EXCLUDED.file_name,
                file_path = EXCLUDED.file_path,
                file_ext = EXCLUDED.file_ext,
                source_type = EXCLUDED.source_type,
                extraction_status = EXCLUDED.extraction_status,
                language = EXCLUDED.language,
                page_count = EXCLUDED.page_count,
                payload = EXCLUDED.payload;
        """)
        copied = cur.rowcount
    conn.commit()
    return copied


# ─────────────────────────────────────────────
# Stats & records (for UI)
# ─────────────────────────────────────────────
def get_stats(conn: psycopg.Connection) -> dict:
    stats: dict[str, Any] = {}
    with conn.cursor() as cur:
        for table in (MAIN_TABLE, POST_TABLE):
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            stats[f"{table}_total"] = cur.fetchone()[0]

        cur.execute(f"SELECT source_type, COUNT(*) FROM {MAIN_TABLE} GROUP BY source_type;")
        stats["by_type"] = dict(cur.fetchall())

        cur.execute(f"SELECT language, COUNT(*) FROM {MAIN_TABLE} GROUP BY language;")
        stats["by_lang"] = dict(cur.fetchall())
    return stats


def get_recent_records(
    conn: psycopg.Connection, table: str = MAIN_TABLE, limit: int = 20
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, doc_id, file_name, source_type, language, page_count, created_at
            FROM {table}
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]