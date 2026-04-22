from __future__ import annotations

from typing import Dict

import dspy

from ..signatures import RouteRetrievalSignature


class RetrieverRouter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(RouteRetrievalSignature)

    def forward(
        self,
        *,
        query: str,
        has_context: bool,
        confidence: str,
        used_web: bool,
    ) -> Dict[str, str]:
        if not has_context:
            return {
                "route": "general",
                "rationale": "No retrieval context is available.",
            }

        prediction = self.predictor(
            query=query,
            has_context=str(bool(has_context)).lower(),
            confidence=confidence,
            used_web=str(bool(used_web)).lower(),
        )
        route = (getattr(prediction, "route", "") or "rag").strip().lower()
        if route not in {"rag", "general"}:
            route = "rag"
        return {
            "route": route,
            "rationale": (getattr(prediction, "rationale", "") or "").strip(),
        }
