from __future__ import annotations

import json
import logging
import re
from typing import Iterator, List, Optional, Union

import requests

from ..dspy_program.modules.rag_pipeline import DSPyRAGPipeline
from .config import RAGConfig
from .schemas import FullDocContext, RetrievedContext

log = logging.getLogger(__name__)


class Synthesizer:
    def __init__(self, config: RAGConfig):
        self.config = config
        if config.llm_provider != "ollama":
            raise ValueError(
                f"Unsupported provider: {config.llm_provider}. "
                "This setup expects LLM_PROVIDER=ollama."
            )
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model_name = config.ollama_model
        self.session = requests.Session()
        self.pipeline = DSPyRAGPipeline(config)

    def warmup(self) -> None:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Reply with one word."},
                {"role": "user", "content": "ready"},
            ],
            "stream": False,
            "keep_alive": self.config.ollama_keep_alive,
            "options": {
                "temperature": 0.0,
                "num_ctx": 256,
                "num_predict": 1,
            },
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(
                    self.config.ollama_connect_timeout_seconds,
                    min(self.config.ollama_request_timeout_seconds, 60),
                ),
            )
            response.raise_for_status()
        except Exception:
            log.warning("Ollama warmup failed.", exc_info=True)

    def generate_response(
        self,
        *,
        query: str,
        contexts: List[RetrievedContext],
        used_web: bool,
        full_docs: Optional[List[FullDocContext]] = None,
        answer_style: Optional[str] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        query = (query or "").strip()
        if not query:
            return "Please provide a question."
        if not contexts and not full_docs:
            return "I could not find relevant information to answer this question."

        try:
            answer = self.pipeline.generate_rag_answer(
                query=query,
                contexts=contexts,
                full_docs=full_docs,
                used_web=used_web,
                answer_style=answer_style or self.config.default_answer_style,
            )
        except Exception as exc:
            raise self._adapt_error(exc) from exc

        if stream:
            return self._pseudo_stream(answer)
        return answer

    def generate_general_response(
        self,
        *,
        query: str,
        answer_style: Optional[str] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        query = (query or "").strip()
        if not query:
            return "Please provide a question."

        try:
            answer = self.pipeline.generate_general_answer(
                query=query,
                answer_style=answer_style or self.config.default_answer_style,
            )
        except Exception as exc:
            raise self._adapt_error(exc) from exc

        if stream:
            return self._pseudo_stream(answer)
        return answer

    def _adapt_error(self, exc: Exception) -> RuntimeError:
        message = str(exc)
        lowered = message.lower()

        if "timed out" in lowered or "timeout" in lowered:
            return RuntimeError(
                f"Ollama timed out while generating from model '{self.model_name}'. "
                "Try again, enable Retrieval only, reduce OLLAMA_NUM_CTX, "
                "or increase the Ollama timeout values in .env."
            )

        if (
            "connection" in lowered
            or "localhost:11434" in lowered
            or "ollama" in lowered and "refused" in lowered
        ):
            return RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama run {self.model_name}"
            )

        return RuntimeError(f"Ollama DSPy error: {message}")

    @staticmethod
    def _pseudo_stream(text: str) -> Iterator[str]:
        for chunk in re.findall(r"\S+\s*", text or ""):
            yield chunk
