from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# AI model name filtering
# ─────────────────────────────────────────────────────────────────────────────

_AI_MODELS = [
    r"chatgpt", r"gpt-4o", r"gpt-4", r"gpt-3", r"gpt",
    r"claude", r"anthropic",
    r"gemini", r"google bard", r"bard",
    r"llama", r"mistral", r"falcon",
    r"openai", r"grok", r"bing chat",
    r"جيميني", r"جيمناي", r"شات جي بي تي", r"شات جي بتي",
]

_AI_PATTERN = re.compile(
    r"\b(" + "|".join(_AI_MODELS) + r")\b",
    flags=re.IGNORECASE,
)

_AI_BLOCK_KEYWORDS = [
    "gemini", "chatgpt", "chat gpt",
    "جيميني", "جيمناي", "شات جي بي تي", "شات جي بتي",
]

_AI_URL_KEYWORDS = [
    "chatgpt", "gpt", "claude", "gemini", "bard",
    "llama", "mistral", "falcon", "openai", "grok", "bing",
]


def remove_ai_models_names(text: str) -> str:
    if not text:
        return ""
    text = _AI_PATTERN.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def remove_links_with_ai(text: str) -> str:
    if not text:
        return ""
    for url in re.findall(r"https?://[^\s]+", text):
        if any(ai in url.lower() for ai in _AI_URL_KEYWORDS):
            text = text.replace(url, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_single_english_letters(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?<!\S)[A-Za-z](?!\S)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def should_remove_text(text: str) -> bool:
    """Return True if the text is dominated by blocked AI keywords."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _AI_BLOCK_KEYWORDS)


def normalize_text(text: str) -> str:
    """
    Safe structural cleaning only — no content removal.
    - Normalise whitespace, line endings, control chars
    - Deduplicate immediately-repeated paragraphs
    """
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u00a0", " ").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    deduped: list[str] = []
    for ln in lines:
        if not deduped or ln != deduped[-1]:
            deduped.append(ln)

    return "\n".join(deduped).strip()


def apply_content_filter(text: str) -> str:
    """
    Remove AI-related content from text:
    - Strip AI model names
    - Remove AI-related URLs
    - Remove lone single English letters
    """
    if not text:
        return ""
    text = remove_ai_models_names(text)
    text = remove_links_with_ai(text)
    text = remove_single_english_letters(text)
    return text


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline — normalize then filter.
    - Remove blocks containing blocked AI keywords (returns empty)
    - Strip AI model names
    - Remove AI-related URLs
    - Remove lone single English letters
    - Normalise whitespace / line endings
    - Deduplicate immediately-repeated paragraphs
    """
    if not text:
        return ""
    if should_remove_text(text):
        return ""
    text = apply_content_filter(text)
    text = normalize_text(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin_chars  = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if arabic_chars > latin_chars and arabic_chars > 20:
        return "ar"
    if latin_chars > arabic_chars and latin_chars > 20:
        return "en"
    if arabic_chars > 0 and latin_chars > 0:
        return "ar-en"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Basic text stats
# ─────────────────────────────────────────────────────────────────────────────

def token_count_simple(text: str) -> int:
    return len(text.split()) if text else 0


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_links(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"https?://[^\s\)\]\>\"\']+", text)


def extract_keywords(text: str, max_keywords: int = 15) -> list[str]:
    if not text:
        return []
    words = re.findall(r"\b[\w\u0600-\u06FF]{3,}\b", text.lower())
    stopwords = {
        "the", "and", "for", "are", "was", "with", "this", "that", "from",
        "have", "has", "had", "not", "but", "they", "will", "can", "all",
        "been", "more", "also", "one", "two", "its", "our", "your", "their",
        "هذا", "هذه", "التي", "الذي", "على", "في", "من", "إلى", "عن",
        "مع", "أو", "أن", "كان", "كانت", "يتم", "حيث", "بما", "كما",
        "لا", "ما", "قد", "هو", "هي", "لم", "لن", "ثم", "بين", "عند",
    }
    freq: dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_keywords]]


def extract_entities(text: str) -> list[str]:
    if not text:
        return []
    en_entities = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    ar_entities = re.findall(r"\b[\u0600-\u06FF]{3,}\b", text)
    combined = en_entities + ar_entities
    freq: dict[str, int] = {}
    for e in combined:
        freq[e] = freq.get(e, 0) + 1
    entities = [e for e, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= 2]
    return entities[:15]


def extract_numbers(text: str) -> list[str]:
    if not text:
        return []
    numbers = re.findall(r"\b\d[\d,\.]*\b", text)
    return list(dict.fromkeys(numbers))[:15]


def extract_dates(text: str) -> list[str]:
    if not text:
        return []
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)"
        r"\s+\d{4}\b",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return list(dict.fromkeys(found))[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Structure inference
# ─────────────────────────────────────────────────────────────────────────────

def _is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line[-1] in {".", ",", ":", "؛", "،"}:
        return False
    if re.search(r"[.،;]{2,}", line):
        return False
    if len(re.findall(r"[\w\u0600-\u06FF]", line)) < 4:
        return False
    return True


def infer_title(text: str) -> str:
    if not text:
        return ""
    for line in text.split("\n"):
        line = line.strip()
        if line and _is_heading_candidate(line):
            return line
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line[:120]
    return ""


def infer_sections(text: str, max_sections: int = 8) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    first = True
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if first:
            first = False
            continue
        if _is_heading_candidate(line) and len(line) < 80:
            result.append(line)
        if len(result) >= max_sections:
            break
    return result


def _detect_chunk_unit(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    heading_count = sum(1 for l in lines if _is_heading_candidate(l))
    if heading_count >= 4:
        return "section"
    avg_len = sum(len(l) for l in lines) / max(len(lines), 1)
    if avg_len > 200:
        return "paragraph"
    return "page"


# ─────────────────────────────────────────────────────────────────────────────
# Main builders
# ─────────────────────────────────────────────────────────────────────────────

def build_structured_json(
    doc_id: str,
    file_name: str,
    source_type: str,
    raw_cleaned_content: str,
) -> dict:
    """
    Build the full structured JSON for a document.
    Includes: title, sections, keywords, entities, dates, numbers, links,
    language, chunking_hints, markdown_text.
    """
    language    = detect_lang(raw_cleaned_content)
    title       = infer_title(raw_cleaned_content)
    sections    = infer_sections(raw_cleaned_content)
    keywords    = extract_keywords(raw_cleaned_content)
    entities    = extract_entities(raw_cleaned_content)
    dates       = extract_dates(raw_cleaned_content)
    numbers     = extract_numbers(raw_cleaned_content)
    links       = extract_links(raw_cleaned_content)

    section    = sections[0] if sections else ""
    subsection = sections[1] if len(sections) > 1 else ""
    section_path: list[str] = []
    if title:
        section_path.append(title)
    section_path.extend(sections)

    chunk_unit    = _detect_chunk_unit(raw_cleaned_content)
    markdown_text = build_markdown_text(raw_cleaned_content)

    return {
        "doc_id":              doc_id,
        "title":               title,
        "section":             section,
        "subsection":          subsection,
        "section_path":        section_path,
        "language":            language,
        "source_type":         source_type,
        "keywords":            keywords,
        "entities":            entities,
        "dates":               dates,
        "numbers":             numbers,
        "links":               links,
        "content":             raw_cleaned_content,
        "raw_cleaned_content": raw_cleaned_content,
        "text_raw":            raw_cleaned_content,   # alias used by viewer tab
        "searchable_text":     raw_cleaned_content,
        "markdown_text":       markdown_text,
        "chunking_hints": {
            "recommended_chunk_unit": chunk_unit,
            "preserve_with_next":     False,
            "preserve_with_previous": False,
        },
    }


def build_markdown_text(raw_cleaned_content: str) -> str:
    """
    Convert cleaned text to lightweight Markdown preserving document structure.
    - First heading-candidate  → H1
    - Subsequent short headings → H2
    - Longer heading candidates → H3
    - Bullet lines kept as-is
    - Everything else → paragraph
    """
    if not raw_cleaned_content:
        return ""

    lines = raw_cleaned_content.split("\n")
    md_lines: list[str] = []
    first_heading_seen = False

    for line in lines:
        line = line.strip()
        if not line:
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            continue

        if re.match(r"^[\-\*\u2022\u25CF]\s+", line):
            md_lines.append(line)
            continue

        if _is_heading_candidate(line):
            if not first_heading_seen:
                md_lines.append(f"# {line}")
                first_heading_seen = True
            elif len(line) < 60:
                md_lines.append(f"## {line}")
            else:
                md_lines.append(f"### {line}")
        else:
            md_lines.append(line)

    # Collapse consecutive blank lines
    result: list[str] = []
    for ln in md_lines:
        if ln == "" and result and result[-1] == "":
            continue
        result.append(ln)

    return "\n\n".join(block for block in "\n".join(result).split("\n\n") if block.strip())