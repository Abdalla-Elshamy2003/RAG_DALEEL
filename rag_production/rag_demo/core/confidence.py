from __future__ import annotations

from typing import List

from .config import RAGConfig
from .schemas import ConfidenceDecision, RetrievedContext


def evaluate_internal_results(
    *,
    contexts: List[RetrievedContext],
    config: RAGConfig,
) -> ConfidenceDecision:
    """
    Decide if internal retrieved documents are enough,
    or if we should fallback to web search.
    """

    if not contexts:
        return ConfidenceDecision(
            should_use_web=True,
            reason="No internal documents were retrieved.",
            confidence="none",
        )

    top_score = contexts[0].rerank_score

    if top_score is None:
        return ConfidenceDecision(
            should_use_web=False,
            reason="Internal results exist, but no reranker score is available.",
            confidence="medium",
        )

    if config.internal_min_rerank_score is not None:
        if top_score < config.internal_min_rerank_score:
            return ConfidenceDecision(
                should_use_web=True,
                reason=(
                    f"Top internal rerank score {top_score:.4f} is below "
                    f"threshold {config.internal_min_rerank_score:.4f}."
                ),
                confidence="low",
            )

    if len(contexts) >= 3:
        confidence = "high"
    else:
        confidence = "medium"

    return ConfidenceDecision(
        should_use_web=False,
        reason="Internal reranked results are available.",
        confidence=confidence,
    )