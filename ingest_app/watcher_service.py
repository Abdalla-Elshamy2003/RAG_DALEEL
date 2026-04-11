"""
watcher_service.py
──────────────────
Background service that watches a folder and auto-ingests new files
into PostgreSQL every POLL_INTERVAL seconds.

Usage:
    python watcher_service.py --folder /data/documents --interval 30

Environment variables (override defaults):
    WATCH_FOLDER     path to watch   (default: ./data)
    POLL_INTERVAL    seconds          (default: 30)
    DB_CONN          postgres DSN     (default: from config.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import psycopg

try:
    from ingest_app.config import AppConfig
    from ingest_app.db import (
        create_tables,
        get_existing_hashes,
        get_hash_by_filepath,
        insert_payload,
        insert_post_processing_payload,
        sync_post_processing_from_main,
    )
    from ingest_app.file_utils import compute_sha256, iter_files
    from ingest_app.payload_builders import (
        build_docx_payload,
        build_pdf_payload,
        build_txt_payload,
    )
except ModuleNotFoundError:
    from config import AppConfig
    from db import (
        create_tables,
        get_existing_hashes,
        get_hash_by_filepath,
        insert_payload,
        insert_post_processing_payload,
        sync_post_processing_from_main,
    )
    from file_utils import compute_sha256, iter_files
    from payload_builders import (
        build_docx_payload,
        build_pdf_payload,
        build_txt_payload,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("watcher.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("watcher")


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

_RUNNING = True


def _handle_signal(sig, _frame):
    global _RUNNING
    log.info("Shutdown signal received (%s) — stopping after current cycle.", sig)
    _RUNNING = False


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─────────────────────────────────────────────────────────────────────────────
# Core scan-and-ingest logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_payload(file_path: Path, file_hash: str):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return build_pdf_payload(file_path, file_hash)
    if ext == ".docx":
        return build_docx_payload(file_path, file_hash)
    if ext == ".txt":
        return build_txt_payload(file_path, file_hash)
    raise ValueError(f"Unsupported extension: {ext}")


def scan_and_ingest(watch_folder: Path, cfg: AppConfig) -> dict:
    """
    Scan watch_folder for new or modified files and ingest/update them.

    Logic:
    - NEW file    (path not in DB)              → insert
    - MODIFIED file (path in DB, hash changed)  → update (upsert)
    - UNCHANGED file (path in DB, same hash)    → skip
    """
    stats = {"found": 0, "inserted": 0, "updated": 0, "skipped": 0, "failed": 0}

    files = list(iter_files(watch_folder))
    stats["found"] = len(files)

    if not files:
        return stats

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            create_tables(conn)

            # 1. Hash all files
            hash_map: dict[Path, str] = {}
            for fp in files:
                try:
                    hash_map[fp] = compute_sha256(fp)
                except Exception as exc:
                    log.warning("Cannot hash %s: %s", fp, exc)
                    stats["failed"] += 1

            # 2. Get existing hash per file_path from DB
            fp_strings = [str(fp).replace("\\", "/") for fp in hash_map]
            known_path_hash = get_hash_by_filepath(conn, fp_strings)

            # 3. Also get existing hashes to avoid treating a renamed file as new
            existing_hashes = get_existing_hashes(conn, list(hash_map.values()))

            for fp, fhash in hash_map.items():
                fp_str = str(fp).replace("\\", "/")
                db_hash = known_path_hash.get(fp_str)

                # UNCHANGED — same path, same hash
                if db_hash == fhash:
                    stats["skipped"] += 1
                    continue

                # MODIFIED — same path, different hash
                is_update = db_hash is not None and db_hash != fhash

                # NEW duplicate hash (different path, same content) — skip
                if not is_update and fhash in existing_hashes:
                    stats["skipped"] += 1
                    continue

                try:
                    payload = _build_payload(fp, fhash)
                    insert_payload(conn, payload)
                    insert_post_processing_payload(conn, payload)
                    conn.commit()
                    if is_update:
                        stats["updated"] += 1
                        log.info("🔄  Updated:  %s", fp.name)
                    else:
                        stats["inserted"] += 1
                        log.info("✅  Inserted: %s", fp.name)
                except Exception as exc:
                    conn.rollback()
                    stats["failed"] += 1
                    log.error("❌  Failed:   %s  →  %s", fp.name, exc)

            # 4. Sync preprocessing → post_processing if anything changed
            if stats["inserted"] + stats["updated"] > 0:
                try:
                    synced = sync_post_processing_from_main(conn)
                    log.info("🔄  Synced %d records to post_processing_data.", synced)
                except Exception as exc:
                    log.warning("Sync warning: %s", exc)

    except Exception as exc:
        log.error("DB connection error: %s", exc)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_watcher(watch_folder: str, interval: int, db_conn: str | None = None):
    folder = Path(watch_folder)
    if not folder.exists():
        log.error("Watch folder does not exist: %s", folder)
        sys.exit(1)

    cfg = AppConfig()
    # Override DB connection if provided
    if db_conn:
        import dataclasses
        cfg = dataclasses.replace(cfg, db_conn=db_conn)

    log.info("=" * 60)
    log.info("  Watcher Service started")
    log.info("  Folder  : %s", folder.resolve())
    log.info("  Interval: %ds", interval)
    log.info("=" * 60)

    cycle = 0
    while _RUNNING:
        cycle += 1
        log.info("── Cycle #%d ─────────────────────────────────", cycle)

        stats = scan_and_ingest(folder, cfg)

        log.info(
            "   found=%d  inserted=%d  updated=%d  skipped=%d  failed=%d",
            stats["found"], stats["inserted"], stats["updated"], stats["skipped"], stats["failed"],
        )

        # Sleep in small chunks so SIGTERM is caught quickly
        for _ in range(interval):
            if not _RUNNING:
                break
            time.sleep(1)

    log.info("Watcher stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Auto-ingest watcher service")
    parser.add_argument(
        "--folder",
        default=os.environ.get("WATCH_FOLDER", "./data"),
        help="Folder to watch (default: ./data or $WATCH_FOLDER)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("POLL_INTERVAL", "30")),
        help="Poll interval in seconds (default: 30 or $POLL_INTERVAL)",
    )
    parser.add_argument(
        "--db-conn",
        default=os.environ.get("DB_CONN"),
        help="Postgres DSN (default: from config.py or $DB_CONN)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_watcher(
        watch_folder=args.folder,
        interval=args.interval,
        db_conn=args.db_conn,
    )