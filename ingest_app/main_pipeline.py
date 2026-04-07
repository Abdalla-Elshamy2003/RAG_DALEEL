from __future__ import annotations

import os
from pathlib import Path

import psycopg

from ingest_app.config import AppConfig
from ingest_app.db import create_table, get_existing_hashes, insert_payload
from ingest_app.file_utils import compute_sha256, iter_files
from ingest_app.payload_builders import build_docx_payload, build_pdf_payload, build_txt_payload


def build_payload(file_path: Path, cfg: AppConfig) -> dict:
    file_hash = compute_sha256(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return build_pdf_payload(file_path, file_hash)
    if ext == ".docx":
        return build_docx_payload(file_path, file_hash)
    if ext == ".txt":
        return build_txt_payload(file_path, file_hash)

    raise ValueError(f"Unsupported file type: {file_path}")


def run_ingestion(root_folder: str | None = None, cfg: AppConfig | None = None) -> int:
    if root_folder is None:
        root_folder = os.getenv("DATA_FOLDER", "./data")
    cfg = cfg or AppConfig()
    root = Path(root_folder)
    if not root.exists():
        print(f"Folder not found: {root}")
        return 1

    files = list(iter_files(root))[:5]  # Limit to first 5 files for testing
    print(f"Found {len(files)} supported files in {root} (limited to 5 for testing)")

    total_files = len(files)
    skipped_files = 0
    inserted_files = 0
    failed_files = 0

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            create_table(conn)

            file_hash_map: dict[Path, str] = {}
            for file_path in files:
                try:
                    file_hash_map[file_path] = compute_sha256(file_path)
                except Exception as exc:
                    failed_files += 1
                    print(f"Failed to hash: {file_path} -> {exc}")

            existing_hashes = get_existing_hashes(conn, list(file_hash_map.values()))

            for file_path, file_hash in file_hash_map.items():
                if file_hash in existing_hashes:
                    print(f"Skipped (already exists): {file_path}")
                    skipped_files += 1
                    continue

                try:
                    payload = build_payload(file_path, cfg)
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
            print(f"Skipped (already exists): {skipped_files}")
            print(f"New files inserted: {inserted_files}")
            print(f"Failed: {failed_files}")

        return 0
    except Exception as exc:
        print(f"Fatal DB error: {exc}")
        return 2


if __name__ == "__main__":
    import sys
    exit_code = run_ingestion()
    sys.exit(exit_code)
