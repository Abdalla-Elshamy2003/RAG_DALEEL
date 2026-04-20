from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv


SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS content_tsv tsvector
GENERATED ALWAYS AS (
    to_tsvector('simple', coalesce(text, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_child_chunks_content_tsv
ON child_chunks USING GIN(content_tsv);

CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_ready
ON child_chunks(embedding_model, embedding_version)
WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_child_chunks_doc_model_version
ON child_chunks(doc_id, embedding_model, embedding_version)
WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_child_chunks_language_expr
ON child_chunks ((metadata->>'doc_language'));

CREATE INDEX IF NOT EXISTS idx_child_chunks_source_type_expr
ON child_chunks ((metadata->>'doc_source_type'));

CREATE INDEX IF NOT EXISTS idx_child_chunks_file_name_expr
ON child_chunks ((metadata->>'doc_file_name'));

CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_hnsw
ON child_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_parent_id
ON parent_chunks(parent_id);
"""


def main() -> None:
    load_dotenv(".env", override=True)

    db_conn = os.getenv("DB_CONN")

    if not db_conn:
        raise RuntimeError("DB_CONN not found in .env")

    with psycopg.connect(db_conn) as conn:
        with conn.cursor() as cur:
            print("Connected to DB.")

            cur.execute("SELECT current_database(), current_schema();")
            print("Current DB/schema:", cur.fetchone())

            print("Applying search indexes...")
            cur.execute(SQL)
            conn.commit()

            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'child_chunks'
                  AND column_name = 'content_tsv';
                """
            )
            print("content_tsv check:", cur.fetchall())

    print("Search indexes fixed successfully.")


if __name__ == "__main__":
    main()