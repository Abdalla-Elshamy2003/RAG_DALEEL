from __future__ import annotations

from typing import Any

import psycopg
from pgvector.psycopg import register_vector

PARENT_TABLE = "parent_chunks"
CHILD_TABLE = "child_chunks"
EMBEDDING_DIM = 1024

VECTOR_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE {PARENT_TABLE}
ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM});

ALTER TABLE {PARENT_TABLE}
ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE {PARENT_TABLE}
ADD COLUMN IF NOT EXISTS embedding_version INT;

ALTER TABLE {PARENT_TABLE}
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

ALTER TABLE {CHILD_TABLE}
ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM});

ALTER TABLE {CHILD_TABLE}
ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE {CHILD_TABLE}
ADD COLUMN IF NOT EXISTS embedding_version INT;

ALTER TABLE {CHILD_TABLE}
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;
"""


def ensure_vector_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(VECTOR_SCHEMA_SQL)
    conn.commit()
    register_vector(conn)


def ensure_hnsw_indexes(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{PARENT_TABLE}_embedding_hnsw
            ON {PARENT_TABLE}
            USING hnsw (embedding vector_cosine_ops)
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{CHILD_TABLE}_embedding_hnsw
            ON {CHILD_TABLE}
            USING hnsw (embedding vector_cosine_ops)
        """)
    conn.commit()


def fetch_rows_needing_embedding(
    conn: psycopg.Connection,
    *,
    table: str,
    id_col: str,
    model_name: str,
    embedding_version: int,
    limit: int,
    only_doc_id: str | None = None,
) -> list[dict[str, Any]]:
    where = [
        "text IS NOT NULL",
        "text <> ''",
        "("
        "embedding IS NULL "
        "OR COALESCE(embedding_model, '') <> %s "
        "OR COALESCE(embedding_version, -1) <> %s"
        ")",
    ]
    params: list[Any] = [model_name, embedding_version]

    if only_doc_id:
        where.append("doc_id = %s")
        params.append(only_doc_id)

    sql = f"""
        SELECT
            {id_col} AS row_id,
            doc_id,
            text
        FROM {table}
        WHERE {' AND '.join(where)}
        ORDER BY id ASC
        LIMIT %s
    """
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def update_embeddings(
    conn: psycopg.Connection,
    *,
    table: str,
    id_col: str,
    rows: list[tuple[str, list[float], str, int]],
) -> None:
    if not rows:
        return

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            UPDATE {table}
            SET
                embedding = %s,
                embedding_model = %s,
                embedding_version = %s,
                embedding_updated_at = now()
            WHERE {id_col} = %s
            """,
            [
                (
                    vector,
                    model_name,
                    version,
                    row_id,
                )
                for row_id, vector, model_name, version in rows
            ],
        )
def get_embedding_stats(conn: psycopg.Connection) -> dict[str, int]:
    stats: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PARENT_TABLE} WHERE embedding IS NOT NULL")
        stats["parent_embedded"] = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {PARENT_TABLE}")
        stats["parent_total"] = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {CHILD_TABLE} WHERE embedding IS NOT NULL")
        stats["child_embedded"] = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {CHILD_TABLE}")
        stats["child_total"] = cur.fetchone()[0]

    return stats
