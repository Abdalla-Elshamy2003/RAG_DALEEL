from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import dspy

if TYPE_CHECKING:
    from ..core.config import RAGConfig


PACKAGE_ROOT = Path(__file__).resolve().parent
SAVED_PROGRAMS_DIR = PACKAGE_ROOT / "saved_programs"


def build_ollama_lm(config: "RAGConfig") -> dspy.LM:
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    return dspy.LM(
        model=f"ollama_chat/{config.ollama_model}",
        model_type="chat",
        api_base=config.ollama_base_url,
        api_key=api_key,
        temperature=0.1,
        max_tokens=config.ollama_num_predict,
        top_p=0.9,
        num_ctx=config.ollama_num_ctx,
        keep_alive=config.ollama_keep_alive,
        timeout=config.ollama_request_timeout_seconds,
        num_retries=2,
        cache=False,
    )


def configure_dspy(config: "RAGConfig") -> dspy.LM:
    return build_ollama_lm(config)


def read_saved_program_metadata(filename: str) -> Dict[str, Any]:
    path = SAVED_PROGRAMS_DIR / filename
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"compiled": True, "path": str(path)}
