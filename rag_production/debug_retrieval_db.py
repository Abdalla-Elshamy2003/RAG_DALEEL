from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(".env", override=True)

    db_conn = os.getenv("DB_CONN")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_version = int(os.getenv("EMBEDDING_VERSION", "1"))

    if not db_conn:
        raise RuntimeError("DB_CONN not found")

    with psycopg.connect(db_conn) as conn:
        with conn.cursor() as cur:
            print("Using filter:")
            print("embedding_model:", embedding_model)
            print("embedding_version:", embedding_version)
            print()

            cur.execute("SELECT COUNT(*) FROM parent_chunks;")
            print("parent_chunks:", cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM child_chunks;")
            print("child_chunks:", cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM child_chunks WHERE embedding IS NOT NULL;")
            print("child_chunks with embedding:", cur.fetchone()[0])

            cur.execute(
                """
                SELECT embedding_model, embedding_version, COUNT(*)
                FROM child_chunks
                GROUP BY embedding_model, embedding_version
                ORDER BY COUNT(*) DESC;
                """
            )
            print("\nEmbedding versions:")
            for row in cur.fetchall():
                print(row)

            cur.execute(
                """
                SELECT COUNT(*)
                FROM child_chunks
                WHERE embedding IS NOT NULL
                  AND embedding_model = %s
                  AND embedding_version = %s;
                """,
                (embedding_model, embedding_version),
            )
            print("\nRows matching current .env filter:", cur.fetchone()[0])

            cur.execute(
                """
                SELECT child_id, doc_id, LEFT(text, 300)
                FROM child_chunks
                WHERE embedding IS NOT NULL
                LIMIT 3;
                """
            )
            print("\nSample embedded child chunks:")
            for row in cur.fetchall():
                print("child_id:", row[0])
                print("doc_id:", row[1])
                print("text:", row[2])
                print("---")


if __name__ == "__main__":
    main()