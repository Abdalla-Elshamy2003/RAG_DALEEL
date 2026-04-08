from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    db_conn: str = os.getenv("DB_CONN", "host=172.16.16.117 port=5433 dbname=docs_ingestion user=esraa password=28102003")
