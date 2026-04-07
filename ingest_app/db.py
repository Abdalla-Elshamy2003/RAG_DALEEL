from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


TABLE_NAME = "file_json_store"


def create_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id BIGSERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT UNIQUE,
            source_type TEXT,
            language TEXT,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_file_json_store_payload_gin
            ON {TABLE_NAME} USING GIN (payload);
        """)

        cur.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_file_json_store_file_hash_unique
            ON {TABLE_NAME}(file_hash);
        """)

    conn.commit()


def get_existing_hashes(conn: psycopg.Connection, hashes: list[str]) -> set[str]:
    if not hashes:
        return set()

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT file_hash
            FROM {TABLE_NAME}
            WHERE file_hash = ANY(%s)
            """,
            (hashes,),
        )
        return {row[0] for row in cur.fetchall() if row and row[0]}


def insert_payload(conn: psycopg.Connection, payload: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {TABLE_NAME} (
                file_name, file_path, file_hash, source_type, language, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_hash)
            DO NOTHING
        """, (
            payload.get("file_name"),
            payload.get("file_path"),
            payload.get("file_hash"),
            payload.get("source_type"),
            payload.get("language"),
            Jsonb(payload),
        ))
