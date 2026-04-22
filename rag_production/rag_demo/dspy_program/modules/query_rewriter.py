from __future__ import annotations

import dspy

from ..signatures import RewriteQuerySignature


class QueryRewriter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(RewriteQuerySignature)

    def forward(self, *, query: str, intent: str, answer_style: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""

        prediction = self.predictor(
            query=query,
            intent=(intent or query).strip(),
            answer_style=answer_style,
        )
        return (getattr(prediction, "rewritten_query", "") or query).strip()
