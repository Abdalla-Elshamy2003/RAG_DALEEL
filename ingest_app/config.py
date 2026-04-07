from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    db_conn: str = "host=172.16.16.117 port=5433 dbname=docs_ingestion user=abdallah password=28102003"
