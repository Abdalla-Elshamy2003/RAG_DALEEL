from __future__ import annotations

from pathlib import Path
import psycopg

try:
    from ingest_app.config import AppConfig
    from ingest_app.db import create_tables, insert_payload
    from ingest_app.file_utils import compute_sha256, iter_files
    from ingest_app.payload_builders import build_docx_payload, build_pdf_payload, build_txt_payload
except ModuleNotFoundError:
    from config import AppConfig
    from db import create_tables, insert_payload
    from file_utils import compute_sha256, iter_files
    from payload_builders import build_docx_payload, build_pdf_payload, build_txt_payload

def build_payload(file_path: Path, file_hash: str) -> dict:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return build_pdf_payload(file_path, file_hash)
    if ext == ".docx":
        return build_docx_payload(file_path, file_hash)
    if ext == ".txt":
        return build_txt_payload(file_path, file_hash)

    raise ValueError(f"Unsupported file type: {file_path}")


def run_ingestion(root_folder: str = "./data", cfg: AppConfig | None = None) -> int:
    cfg = cfg or AppConfig()
    root = Path(root_folder)
    if not root.exists():
        print(f"Folder not found: {root}")
        return 1

    files = list(iter_files(root))
    print(f"Found {len(files)} supported files in {root}")

    total_files = len(files)
    inserted_files = 0
    failed_files = 0

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            create_tables(conn)

            for file_path in files:
                try:
                    file_hash = compute_sha256(file_path)
                    payload = build_payload(file_path, file_hash)
                    insert_payload(conn, payload)
                    conn.commit()
                    print(f"Inserted: {file_path}")
                    inserted_files += 1
                except Exception as exc:
                    conn.rollback()
                    failed_files += 1
                    print(f"Failed: {file_path} -> {exc}")

            print("\n=== SUMMARY ===")
            print(f"Total files found: {total_files}")
            print(f"Inserted: {inserted_files}")
            print(f"Failed: {failed_files}")

        return 0
    except Exception as exc:
        print(f"Fatal DB error: {exc}")
        return 2