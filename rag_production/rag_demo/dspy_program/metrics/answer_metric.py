from __future__ import annotations

import re


def _normalize(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return [token for token in re.split(r"[^a-z0-9\u0600-\u06ff]+", text) if token]


def answer_metric(example, prediction, trace=None) -> float:
    reference = getattr(example, "answer", "") or getattr(example, "reference_answer", "")
    candidate = getattr(prediction, "answer", "") or ""
    reference_tokens = set(_normalize(reference))
    candidate_tokens = set(_normalize(candidate))
    if not reference_tokens:
        return 0.0
    overlap = len(reference_tokens & candidate_tokens)
    return overlap / len(reference_tokens)
