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
            CREATE INDEX IF NOT EXISTS idx_{table}_file_hash
                ON {table}(file_hash);
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_file_path
                ON {table}(file_path);
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
    """Simple INSERT - allows duplicate file_path entries."""
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {table} (
                doc_id, file_name, file_path, file_ext, file_hash,
                source_type, extraction_status, language, page_count, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    Copy all rows from preprocessing_data to post_processing_data.
    Uses upsert semantics so repeated sync updates existing rows.
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
            FROM {MAIN_TABLE};
            ON CONFLICT (doc_id) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                file_path = EXCLUDED.file_path,
                file_ext = EXCLUDED.file_ext,
                file_hash = EXCLUDED.file_hash,
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


# ─────────────────────────────────────────────
# Chunking helpers
# ─────────────────────────────────────────────
def get_unprocessed_docs_for_chunking(
    conn: psycopg.Connection,
    limit: int = 100,
    source_table: str = MAIN_TABLE
) -> list[dict]:
    """Get documents that haven't been chunked yet."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT d.doc_id, d.payload, d.source_type
            FROM {source_table} d
            WHERE NOT EXISTS (
                SELECT 1 FROM chunks c
                WHERE c.doc_id = d.doc_id
            )
            ORDER BY d.created_at DESC
            LIMIT %s
        """, (limit,))

        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def save_chunks_to_db(conn: psycopg.Connection, chunks: list[dict]) -> int:
    """Save chunks to database. Returns number of chunks saved."""
    if not chunks:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute("""
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, chunk_text,
                    char_count, token_count, embedding_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    char_count = EXCLUDED.char_count,
                    token_count = EXCLUDED.token_count
            """, (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["chunk_index"],
                chunk["chunk_text"],
                chunk["char_count"],
                chunk["token_count"],
                chunk.get("embedding_id"),
            ))
            inserted += 1

    conn.commit()
    return inserted


def get_chunks_for_embedding(
    conn: psycopg.Connection,
    limit: int = 1000
) -> list[dict]:
    """Get chunks that don't have embeddings yet."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, doc_id, chunk_text
            FROM chunks
            WHERE embedding_id IS NULL
            ORDER BY created_at ASC
            LIMIT %s
        """, (limit,))

        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ─────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────
def save_embeddings_to_db(conn: psycopg.Connection, chunks_with_embeddings: list[dict]) -> int:
    """
    Save embeddings to the database.
    Returns number of embeddings saved.
    """
    if not chunks_with_embeddings:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for chunk in chunks_with_embeddings:
            if chunk.get("embedding_vector") is None:
                continue

            cur.execute("""
                INSERT INTO embeddings (
                    embedding_id, chunk_id, model_name, dimensions, embedding_vector
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (embedding_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    dimensions = EXCLUDED.dimensions,
                    embedding_vector = EXCLUDED.embedding_vector
            """, (
                chunk["embedding_id"],
                chunk["chunk_id"],
                chunk["model_name"],
                chunk["dimensions"],
                chunk["embedding_vector"],
            ))

            # Update chunk with embedding reference
            cur.execute("""
                UPDATE chunks
                SET embedding_id = %s
                WHERE chunk_id = %s
            """, (chunk["embedding_id"], chunk["chunk_id"]))

            inserted += 1

    conn.commit()
    return inserted


def search_similar_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    model_name: str,
    limit: int = 10,
    min_similarity: float = 0.7
) -> list[dict]:
    """
    Search for similar chunks using cosine similarity.
    Returns chunks ordered by similarity score.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.chunk_text,
                c.chunk_index,
                e.model_name,
                1 - (e.embedding_vector <=> %s::vector) as similarity
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE e.model_name = %s
            AND 1 - (e.embedding_vector <=> %s::vector) >= %s
            ORDER BY e.embedding_vector <=> %s::vector
            LIMIT %s
        """, (query_embedding, model_name, query_embedding, min_similarity, limit))

        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]