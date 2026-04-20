from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(".env", override=True)

    db_conn = os.getenv("DB_CONN")

    if not db_conn:
        raise RuntimeError("DB_CONN not found in .env")

    with psycopg.connect(db_conn) as conn:
        with conn.cursor() as cur:
            print("DB connected successfully")

            # 1. Check content_tsv column
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'child_chunks'
                  AND column_name = 'content_tsv';
                """
            )
            print("content_tsv check:", cur.fetchall())

            # 2. Check embedding model/version
            cur.execute(
                """
                SELECT embedding_model, embedding_version, COUNT(*)
                FROM child_chunks
                GROUP BY embedding_model, embedding_version
                ORDER BY embedding_model, embedding_version;
                """
            )
            print("embedding versions:")
            for row in cur.fetchall():
                print(row)

            # 3. Count chunks
            cur.execute("SELECT COUNT(*) FROM parent_chunks;")
            parent_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM child_chunks;")
            child_count = cur.fetchone()[0]

            print("parent_chunks:", parent_count)
            print("child_chunks:", child_count)


if __name__ == "__main__":
    main()