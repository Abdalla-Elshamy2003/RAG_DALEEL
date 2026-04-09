from __future__ import annotations

import re

def remove_ai_models_names(text: str) -> str:
    if not text:
        return ""
    ai_models = [
        r"chatgpt", r"gpt-3", r"gpt-4", r"gpt-4o", r"gpt",
        r"claude", r"anthropic",
        r"gemini", r"google bard", r"bard",
        r"llama", r"mistral", r"falcon",
        r"openai", r"grok", r"bing chat",
        r"جيميني", r"جيمناي", r"شات جي بي تي", r"شات جي بتي"
    ]

    pattern = r"\b(" + "|".join(ai_models) + r")\b"
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def remove_links_with_ai(text: str) -> str:
    if not text:
        return ""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    for url in urls:
        if any(ai in url.lower() for ai in [
            "chatgpt", "gpt", "claude", "gemini", "bard", "llama", "mistral", "falcon", "openai", "grok", "bing"
        ]):
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
    if not text:
        return False
    ai_keywords = [
        "gemini", "chatgpt", "chat gpt", "جيميني", "جيمناي", "شات جي بي تي", "شات جي بتي"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ai_keywords)


def clean_text(text: str) -> str:
    if not text:
        return ""
    if should_remove_text(text):
        return ""
    text = remove_ai_models_names(text)
    text = remove_links_with_ai(text)
    text = remove_single_english_letters(text)
    text = text.replace("\x00", " ").replace("\u00a0", " ").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def detect_lang(text: str) -> str:
    if not text:
        return "unknown"
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if arabic_chars > latin_chars and arabic_chars > 20:
        return "ar"
    if latin_chars > arabic_chars and latin_chars > 20:
        return "en"
    return "unknown"


def token_count_simple(text: str) -> int:
    return len(text.split()) if text else 0


def extract_links(text: str) -> list[str]:
    if not text:
        return []
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    return urls


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    if not text:
        return []
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 2:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def extract_entities(text: str) -> list[str]:
    # Simple entity extraction: capitalized words or Arabic names
    if not text:
        return []
    # For simplicity, extract capitalized words as potential entities
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)  # English capitalized
    arabic_entities = re.findall(r'\b[\u0600-\u06FF]{2,}\b', text)  # Arabic words
    return list(set(entities + arabic_entities))[:10]  # Limit to 10


def extract_numbers(text: str) -> list[str]:
    if not text:
        return []
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    return numbers[:10]


def extract_dates(text: str) -> list[str]:
    if not text:
        return []
    # Simple date patterns
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2} \w+ \d{4}\b'
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return list(set(dates))[:5]


def infer_title(text: str) -> str:
    if not text:
        return ""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) < 100:  # Assume title is short
            return line
    return lines[0].strip() if lines else ""


def infer_sections(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split('\n')
    sections = []
    for line in lines:
        line = line.strip()
        if line and len(line) < 50 and not line.endswith('.'):  # Potential section
            sections.append(line)
    return sections[:5]


def build_structured_json(doc_id: str, file_name: str, source_type: str, raw_cleaned_content: str) -> dict:
    language = detect_lang(raw_cleaned_content)
    title = infer_title(raw_cleaned_content)
    sections = infer_sections(raw_cleaned_content)
    keywords = extract_keywords(raw_cleaned_content)
    entities = extract_entities(raw_cleaned_content)
    dates = extract_dates(raw_cleaned_content)
    numbers = extract_numbers(raw_cleaned_content)
    links = extract_links(raw_cleaned_content)
    
    # For simplicity, set section to first section if any
    section = sections[0] if sections else ""
    subsection = sections[1] if len(sections) > 1 else ""
    section_path = [title] + sections if title else sections
    
    markdown_text = build_markdown_text(raw_cleaned_content)
    
    return {
        "doc_id": doc_id,
        "title": title,
        "section": section,
        "subsection": subsection,
        "section_path": section_path,
        "language": language,
        "source_type": source_type,
        "keywords": keywords,
        "entities": entities,
        "dates": dates,
        "numbers": numbers,
        "link": links,
        "text_raw": raw_cleaned_content,  # All text content
        "searchable_text": raw_cleaned_content,  # For now, same as raw
        "markdown_text": markdown_text
    }


def build_markdown_text(raw_cleaned_content: str) -> str:
    if not raw_cleaned_content:
        return ""
    lines = raw_cleaned_content.split('\n')
    markdown_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 50 and not line.endswith('.'):
            # Potential heading
            markdown_lines.append(f"## {line}")
        else:
            markdown_lines.append(line)
    return '\n\n'.join(markdown_lines)
