from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")
_LONG_REPEAT_RE = re.compile(r"(.)\1{2,}")

_ARABIC_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ـ": "",
    }
)


@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    normalized: str
    retrieval_query: str
    keyword_query: str
    language_hint: str


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(_ARABIC_CHAR_MAP)
    text = _LONG_REPEAT_RE.sub(r"\1\1", text)
    text = _NON_WORD_RE.sub(" ", text.lower())
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def _detect_language_hint(text: str) -> str:
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    has_english = bool(re.search(r"[A-Za-z]", text))

    if has_arabic and has_english:
        return "mixed"
    if has_arabic:
        return "arabic"
    if has_english:
        return "english"
    return "unknown"


def process_query(query: str) -> ProcessedQuery:
    original = (query or "").strip()
    normalized = _normalize_text(original)

    tokens: list[str] = []
    for token in normalized.split():
        if len(token) <= 1:
            continue
        if token not in tokens:
            tokens.append(token)

    keyword_query = " OR ".join(tokens[:12]) if tokens else normalized
    retrieval_query = normalized or original

    return ProcessedQuery(
        original=original,
        normalized=normalized or original,
        retrieval_query=retrieval_query,
        keyword_query=keyword_query or retrieval_query,
        language_hint=_detect_language_hint(original or normalized),
    )
