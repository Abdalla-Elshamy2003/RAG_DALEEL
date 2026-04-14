from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    db_conn: str = field(default_factory=lambda: os.environ.get("DB_CONN", ""))

    def __post_init__(self):
        if not self.db_conn:
            raise ValueError("DB_CONN environment variable is not set. Please create a .env file.")
