from __future__ import annotations

import re
from typing import Dict

import dspy

from ..signatures import AnalyzeQuerySignature


class QueryAnalyzer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.ChainOfThought(AnalyzeQuerySignature)

    def forward(self, *, query: str, answer_style: str) -> Dict[str, str]:
        query = (query or "").strip()
        if not query:
            return {
                "intent": "",
                "retrieval_query": "",
                "language_instruction": "Answer in the same language as the user's question.",
            }

        prediction = self.predictor(query=query, answer_style=answer_style)
        language_instruction = (
            getattr(prediction, "language_instruction", "") or ""
        ).strip()
        if self._contains_arabic(query):
            language_instruction = (
                "Answer only in natural, professional Arabic. "
                "Do not answer in English, except for necessary technical terms or proper nouns."
            )
        elif self._contains_english(query):
            language_instruction = (
                "Answer only in clear, natural English."
            )
        elif not language_instruction:
            language_instruction = "Answer in the same language as the user's question."

        return {
            "intent": (getattr(prediction, "intent", "") or query).strip(),
            "retrieval_query": (
                getattr(prediction, "retrieval_query", "") or query
            ).strip(),
            "language_instruction": language_instruction,
        }

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text or ""))

    @staticmethod
    def _contains_english(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text or ""))
