from __future__ import annotations

import dspy

from ..signatures import GenerateGeneralAnswerSignature, GenerateRAGAnswerSignature


class AnswerGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rag_predictor = dspy.Predict(GenerateRAGAnswerSignature)
        self.general_predictor = dspy.Predict(GenerateGeneralAnswerSignature)

    def generate_rag(
        self,
        *,
        question: str,
        rewritten_question: str,
        answer_style: str,
        source_scope: str,
        language_instruction: str,
        citation_style: str,
        context_block: str,
    ) -> str:
        prediction = self.rag_predictor(
            question=question,
            rewritten_question=rewritten_question,
            answer_style=answer_style,
            source_scope=source_scope,
            language_instruction=language_instruction,
            citation_style=citation_style,
            context_block=context_block,
        )
        return (getattr(prediction, "answer", "") or "").strip()

    def generate_general(
        self,
        *,
        question: str,
        rewritten_question: str,
        answer_style: str,
        language_instruction: str,
    ) -> str:
        prediction = self.general_predictor(
            question=question,
            rewritten_question=rewritten_question,
            answer_style=answer_style,
            language_instruction=language_instruction,
        )
        return (getattr(prediction, "answer", "") or "").strip()
