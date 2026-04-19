from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def main() -> None:
    # Load .env from the current project folder
    load_dotenv()

    db_conn = os.getenv("DB_CONN")

    if not db_conn:
        raise RuntimeError(
            "DB_CONN not found. Make sure .env exists in the project root "
            "and contains DB_CONN."
        )

    migration_path = Path("migrations") / "rag.sql"

    if not migration_path.exists():
        raise FileNotFoundError(
            f"Migration file not found: {migration_path}. "
            "Make sure your SQL file is here: migrations/rag.sql"
        )

    sql = migration_path.read_text(encoding="utf-8")

    print(f"Running migration: {migration_path}")

    with psycopg.connect(db_conn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

    print("Migration applied successfully.")


if __name__ == "__main__":
    main()