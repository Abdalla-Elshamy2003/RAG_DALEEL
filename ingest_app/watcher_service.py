"""
watcher_service.py
──────────────────
Background service that watches a folder and auto-ingests new files
into PostgreSQL every POLL_INTERVAL seconds.

Responsibility: write to ``preprocessing_data`` ONLY.
``post_processing_data`` is left for a separate detached job.

Usage:
    python watcher_service.py --folder /data/documents --interval 30

Environment variables (override defaults):
    WATCH_FOLDER     path to watch   (default: ./data)
    POLL_INTERVAL    seconds          (default: 30)
    DB_CONN          postgres DSN     (default: from config.py)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg

try:
    from ingest_app.config import AppConfig
    from ingest_app.db import (
        create_tables,
        insert_payload,
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
        insert_payload,
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


# Cache to store mtime of files to avoid redundant hashing
_MTIME_CACHE: dict[Path, float] = {}


def scan_and_ingest(watch_folder: Path, cfg: AppConfig) -> dict:
    """
    Scan watch_folder and INSERT all files to DB.
    Allows duplicates - same file will create multiple records.
    Uses mtime cache to avoid reprocessing unchanged files in same session.
    """
    stats = {"found": 0, "inserted": 0, "skipped": 0, "failed": 0}

    if not watch_folder.exists():
        log.warning("Watch folder missing: %s. Skipping this cycle.", watch_folder)
        return stats

    files = list(iter_files(watch_folder))
    stats["found"] = len(files)

    if not files:
        return stats

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            create_tables(conn)

            # 1. mtime check — skip files unchanged since last cycle
            to_process: list[Path] = []
            for fp in files:
                try:
                    mtime = fp.stat().st_mtime
                    if _MTIME_CACHE.get(fp) == mtime:
                        continue
                    to_process.append(fp)
                except Exception as exc:
                    log.warning("Cannot check file %s: %s", fp, exc)
                    stats["failed"] += 1

            if not to_process:
                stats["skipped"] = stats["found"] - stats["failed"]
                return stats

            # 2. Parallel payload extraction via ThreadPoolExecutor
            processed_payloads: list[tuple[Path, dict]] = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(_build_payload, fp, compute_sha256(fp)): fp
                    for fp in to_process
                }

                for future in as_completed(future_to_file):
                    fp = future_to_file[future]
                    try:
                        payload = future.result()
                        processed_payloads.append((fp, payload))
                    except Exception as exc:
                        stats["failed"] += 1
                        log.error("Extraction failed: %s -> %s", fp.name, exc)

            # 3. Sequential DB insertion
            for fp, payload in processed_payloads:
                try:
                    insert_payload(conn, payload)
                    conn.commit()

                    _MTIME_CACHE[fp] = fp.stat().st_mtime
                    stats["inserted"] += 1
                    log.info("Inserted: %s", fp.name)
                except Exception as exc:
                    conn.rollback()
                    stats["failed"] += 1
                    log.error("DB failed: %s -> %s", fp.name, exc)

    except Exception as exc:
        log.error("DB connection error: %s", exc)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_watcher(watch_folder: str, interval: int, db_conn: str | None = None):
    folder = Path(watch_folder)

    # Create folder if it doesn't exist (especially for default ./data)
    if not folder.exists():
        log.info("Watch folder %s not found. Attempting to create...", folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("Could not create folder: %s. Watcher will wait for it to appear.", exc)

    cfg = AppConfig()
    if db_conn:
        cfg = dataclasses.replace(cfg, db_conn=db_conn)

    log.info("=" * 60)
    log.info("  Watcher Service started")
    log.info("  Folder  : %s", folder.resolve())
    log.info("  Interval: %ds", interval)
    log.info("=" * 60)

    cycle = 0
    while _RUNNING:
        cycle += 1
        log.info("-- Cycle #%d --", cycle)

        stats = scan_and_ingest(folder, cfg)

        log.info(
            "   found=%d  inserted=%d  updated=%d  skipped=%d  failed=%d",
            stats["found"], stats["inserted"], stats["updated"],
            stats["skipped"], stats["failed"],
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