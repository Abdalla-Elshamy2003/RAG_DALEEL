from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

try:
    from ingest_app.db import POST_TABLE
except ModuleNotFoundError:
    from db import POST_TABLE

SOURCE_TABLE = POST_TABLE
PARENT_TABLE = "parent_chunks"
CHILD_TABLE = "child_chunks"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {PARENT_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    parent_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL REFERENCES {POST_TABLE}(doc_id) ON DELETE CASCADE,
    parent_index INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    char_count INT NOT NULL,
    start_char INT,
    end_char INT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, parent_index)
);

CREATE TABLE IF NOT EXISTS {CHILD_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    child_id TEXT NOT NULL UNIQUE,
    parent_id TEXT NOT NULL REFERENCES {PARENT_TABLE}(parent_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES {POST_TABLE}(doc_id) ON DELETE CASCADE,
    child_index INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    char_count INT NOT NULL,
    start_char INT,
    end_char INT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, child_index)
);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc_id ON {PARENT_TABLE}(doc_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_doc_id ON {CHILD_TABLE}(doc_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id ON {CHILD_TABLE}(parent_id);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_metadata_gin ON {PARENT_TABLE} USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_child_chunks_metadata_gin ON {CHILD_TABLE} USING GIN(metadata);
"""


def create_chunk_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def fetch_documents_needing_chunking(
    conn: psycopg.Connection,
    *,
    strategy_name: str,
    strategy_version: int,
    tokenizer_model: str,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    min_child_chunk_tokens: int,
    limit: int | None = None,
    only_doc_id: str | None = None,
    only_hash: str | None = None,
) -> list[dict[str, Any]]:
    clauses = ["COALESCE(s.payload->>'raw_cleaned_content', s.payload->>'content', '') <> ''"]
    params: list[Any] = []

    if only_doc_id:
        clauses.append("s.doc_id = %s")
        params.append(only_doc_id)

    if only_hash:
        clauses.append("s.file_hash = %s")
        params.append(only_hash)

    sql = f"""
    WITH parent_info AS (
        SELECT
            doc_id,
            COUNT(*) AS parent_count,
            MAX(metadata->>'doc_file_hash') AS chunk_file_hash,
            MAX(metadata->>'strategy_name') AS chunk_strategy_name,
            MAX(NULLIF(metadata->>'strategy_version', '')::int) AS chunk_strategy_version,
            MAX(metadata->>'tokenizer_model') AS chunk_tokenizer_model,
            MAX(NULLIF(metadata->>'parent_chunk_size', '')::int) AS chunk_parent_chunk_size,
            MAX(NULLIF(metadata->>'parent_chunk_overlap', '')::int) AS chunk_parent_chunk_overlap,
            MAX(NULLIF(metadata->>'child_chunk_size', '')::int) AS chunk_child_chunk_size,
            MAX(NULLIF(metadata->>'child_chunk_overlap', '')::int) AS chunk_child_chunk_overlap,
            MAX(NULLIF(metadata->>'min_child_chunk_tokens', '')::int) AS chunk_min_child_chunk_tokens
        FROM {PARENT_TABLE}
        GROUP BY doc_id
    ),
    child_info AS (
        SELECT
            doc_id,
            COUNT(*) AS child_count
        FROM {CHILD_TABLE}
        GROUP BY doc_id
    )
    SELECT
        s.doc_id,
        s.file_name,
        s.file_path,
        s.file_hash,
        s.source_type,
        s.extraction_status,
        s.language,
        s.page_count,
        COALESCE(s.payload->>'raw_cleaned_content', s.payload->>'content', '') AS raw_cleaned_content
    FROM {SOURCE_TABLE} s
    LEFT JOIN parent_info p ON p.doc_id = s.doc_id
    LEFT JOIN child_info c ON c.doc_id = s.doc_id
    WHERE {" AND ".join(clauses)}
      AND (
            p.doc_id IS NULL
         OR c.doc_id IS NULL
         OR COALESCE(p.parent_count, 0) = 0
         OR COALESCE(c.child_count, 0) = 0
         OR COALESCE(p.chunk_file_hash, '') <> COALESCE(s.file_hash, '')
         OR COALESCE(p.chunk_strategy_name, '') <> %s
         OR COALESCE(p.chunk_strategy_version, -1) <> %s
         OR COALESCE(p.chunk_tokenizer_model, '') <> %s
         OR COALESCE(p.chunk_parent_chunk_size, -1) <> %s
         OR COALESCE(p.chunk_parent_chunk_overlap, -1) <> %s
         OR COALESCE(p.chunk_child_chunk_size, -1) <> %s
         OR COALESCE(p.chunk_child_chunk_overlap, -1) <> %s
         OR COALESCE(p.chunk_min_child_chunk_tokens, -1) <> %s
      )
    ORDER BY s.created_at ASC
    """

    params.extend([
        strategy_name,
        strategy_version,
        tokenizer_model,
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
        min_child_chunk_tokens,
    ])

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def delete_chunks_for_doc(conn: psycopg.Connection, doc_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {CHILD_TABLE} WHERE doc_id = %s", (doc_id,))
        cur.execute(f"DELETE FROM {PARENT_TABLE} WHERE doc_id = %s", (doc_id,))


def insert_parent_chunks(conn: psycopg.Connection, parents: list[Any]) -> None:
    if not parents:
        return

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {PARENT_TABLE} (
                parent_id, doc_id, parent_index, text,
                token_count, char_count, start_char, end_char, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    p.parent_id,
                    p.doc_id,
                    p.parent_index,
                    p.text,
                    p.token_count,
                    p.char_count,
                    p.start_char,
                    p.end_char,
                    Jsonb(p.metadata),
                )
                for p in parents
            ],
        )


def insert_child_chunks(conn: psycopg.Connection, children: list[Any]) -> None:
    if not children:
        return

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {CHILD_TABLE} (
                child_id, parent_id, doc_id, child_index, text,
                token_count, char_count, start_char, end_char, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    c.child_id,
                    c.parent_id,
                    c.doc_id,
                    c.child_index,
                    c.text,
                    c.token_count,
                    c.char_count,
                    c.start_char,
                    c.end_char,
                    Jsonb(c.metadata),
                )
                for c in children
            ],
        )


def get_chunk_stats(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PARENT_TABLE}")
        parent_total = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {CHILD_TABLE}")
        child_total = cur.fetchone()[0]

    return {"parent_total": parent_total, "child_total": child_total}