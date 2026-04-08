from __future__ import annotations

import re

def remove_ai_models_names(text:str)->str:
    if not text:
        return ""
    ai_models=[
        r"chatgpt", r"gpt-3", r"gpt-4", r"gpt-4o", r"gpt", 
        r"claude", r"anthropic", 
        r"gemini", r"google bard", r"bard",
        r"llama", r"mistral", r"falcon",
        r"openai", r"grok", r"bing chat"
    ]

    pattern = pattern = r"\b(" + "|".join(ai_models) + r")\b"
    text = re.sub(pattern," ",text,flags= re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()



def clean_text(text: str) -> str:
    if not text:
        return ""
    text = remove_ai_models_names(text)
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
    if arabic_chars >latin_chars and arabic_chars > 20:
        return "ar"
    if latin_chars > arabic_chars and latin_chars > 20:
        return "en"
    return "unknown"


def token_count_simple(text: str) -> int:
    return len(text.split()) if text else 0
