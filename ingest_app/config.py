from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    db_conn: str = "host=localhost port=5433 dbname=docs_ingestion user=postgres password=28102003"
