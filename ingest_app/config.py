from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    db_conn: str = "host=135.181.117.17 port=5432 dbname=docs_ingestion user=abdu password=Neurix@123!@#"
