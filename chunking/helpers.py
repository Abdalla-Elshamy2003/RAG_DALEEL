from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def normalize_text(text: str) -> str:
    return (
        (text or "")
        .replace("\\ufeff", "")
        .replace("\\x00", " ")
        .replace("\\r\\n", "\\n")
        .replace("\\r", "\\n")
        .strip()
    )


def smart_concat(left: str, right: str) -> str:
    left = normalize_text(left)
    right = normalize_text(right)

    if not left:
        return right
    if not right:
        return left
    if right in left:
        return left

    max_overlap = min(len(left), len(right), 300)
    overlap = 0

    for k in range(max_overlap, 19, -1):
        if left[-k:] == right[:k]:
            overlap = k
            break

    if overlap > 0:
        return (left + right[overlap:]).strip()

    return f"{left}\\n\\n{right}".strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        x = int(value)
        return x if x >= 0 else None
    except Exception:
        return None


def stable_id(
    prefix: str,
    doc_id: str,
    index: int,
    start_char: int | None,
    end_char: int | None,
) -> str:
    if start_char is None or end_char is None:
        return f"{prefix}_{doc_id}_{index}"
    return f"{prefix}_{doc_id}_{index}_{start_char}_{end_char}"