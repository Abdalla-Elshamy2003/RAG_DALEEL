from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParentRow:
    parent_id: str
    doc_id: str
    parent_index: int
    text: str
    token_count: int
    char_count: int
    start_char: int | None
    end_char: int | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChildRow:
    child_id: str
    parent_id: str
    doc_id: str
    child_index: int
    text: str
    token_count: int
    char_count: int
    start_char: int | None
    end_char: int | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class TempChild:
    text: str
    token_count: int
    char_count: int
    start_char: int | None
    end_char: int | None