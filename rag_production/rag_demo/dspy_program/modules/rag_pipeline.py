from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

import dspy

from ..settings import configure_dspy, read_saved_program_metadata
from .answer_generator import AnswerGenerator
from .context_builder import ContextBuilder
from .query_analyzer import QueryAnalyzer
from .query_rewriter import QueryRewriter
from .retriever_router import RetrieverRouter

if TYPE_CHECKING:
    from ...core.config import RAGConfig
    from ...core.schemas import FullDocContext, RetrievedContext


class DSPyRAGPipeline:
    def __init__(self, config: "RAGConfig") -> None:
        self.config = config
        self.lm = configure_dspy(config)
        self.query_analyzer = QueryAnalyzer()
        self.query_rewriter = QueryRewriter()
        self.retriever_router = RetrieverRouter()
        self.context_builder = ContextBuilder(config)
        self.answer_generator = AnswerGenerator()
        self.saved_programs = {
            "rag": read_saved_program_metadata("compiled_rag.json"),
            "router": read_saved_program_metadata("compiled_router.json"),
        }

    @staticmethod
    def _confidence_label(contexts: List["RetrievedContext"]) -> str:
        if not contexts:
            return "none"

        top_score = contexts[0].rerank_score
        if top_score is None:
            return "low"
        if top_score >= 0.75:
            return "high"
        if top_score >= 0.3:
            return "medium"
        return "low"

    def generate_rag_answer(
        self,
        *,
        query: str,
        contexts: List["RetrievedContext"],
        full_docs: Optional[List["FullDocContext"]],
        used_web: bool,
        answer_style: str,
    ) -> str:
        with dspy.settings.context(lm=self.lm):
            analysis = self.query_analyzer(query=query, answer_style=answer_style)
            rewritten_query = self.query_rewriter(
                query=query,
                intent=analysis["intent"],
                answer_style=answer_style,
            )
            built_context = self.context_builder.build(
                contexts=contexts,
                full_docs=full_docs,
                used_web=used_web,
            )
            route = self.retriever_router(
                query=query,
                has_context=bool(contexts or full_docs),
                confidence=self._confidence_label(contexts),
                used_web=used_web,
            )
            if route["route"] == "general" and not (contexts or full_docs):
                answer = self.answer_generator.generate_general(
                    question=query,
                    rewritten_question=rewritten_query or query,
                    answer_style=answer_style,
                    language_instruction=analysis["language_instruction"],
                )
                return self._enforce_language_alignment(
                    query=query,
                    answer=answer,
                    regenerate=lambda stronger_instruction: self.answer_generator.generate_general(
                        question=query,
                        rewritten_question=rewritten_query or query,
                        answer_style=answer_style,
                        language_instruction=stronger_instruction,
                    ),
                )

            answer = self.answer_generator.generate_rag(
                question=query,
                rewritten_question=rewritten_query or query,
                answer_style=answer_style,
                source_scope=built_context.source_scope,
                language_instruction=analysis["language_instruction"],
                citation_style=built_context.citation_style,
                context_block=built_context.context_block,
            )
            return self._enforce_language_alignment(
                query=query,
                answer=answer,
                regenerate=lambda stronger_instruction: self.answer_generator.generate_rag(
                    question=query,
                    rewritten_question=rewritten_query or query,
                    answer_style=answer_style,
                    source_scope=built_context.source_scope,
                    language_instruction=stronger_instruction,
                    citation_style=built_context.citation_style,
                    context_block=built_context.context_block,
                ),
            )

    def generate_general_answer(
        self,
        *,
        query: str,
        answer_style: str,
    ) -> str:
        with dspy.settings.context(lm=self.lm):
            analysis = self.query_analyzer(query=query, answer_style=answer_style)
            rewritten_query = self.query_rewriter(
                query=query,
                intent=analysis["intent"],
                answer_style=answer_style,
            )
            answer = self.answer_generator.generate_general(
                question=query,
                rewritten_question=rewritten_query or query,
                answer_style=answer_style,
                language_instruction=analysis["language_instruction"],
            )
            return self._enforce_language_alignment(
                query=query,
                answer=answer,
                regenerate=lambda stronger_instruction: self.answer_generator.generate_general(
                    question=query,
                    rewritten_question=rewritten_query or query,
                    answer_style=answer_style,
                    language_instruction=stronger_instruction,
                ),
            )

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text or ""))

    def _enforce_language_alignment(self, *, query: str, answer: str, regenerate) -> str:
        if self._contains_arabic(query) and not self._contains_arabic(answer):
            stronger_instruction = (
                "Answer only in natural, fluent Arabic. "
                "Do not answer in English. Keep English only for necessary technical terms or proper nouns. "
                "Make the answer detailed, warm, polished, and human-sounding."
            )
            regenerated = regenerate(stronger_instruction)
            if regenerated:
                return regenerated
        return answer
