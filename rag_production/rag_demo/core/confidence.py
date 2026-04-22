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
    Evaluate whether internal reranked results are sufficient.

    bge-reranker-v2-m3 produces raw logits (roughly -10 to +10).
    We use the top score + margin (gap to second result) as the signal.

    Thresholds to calibrate after running ~50 real queries:
      - LOW_THRESHOLD:    top score below this → definitely use web
      - HIGH_THRESHOLD:   top score above this → strong internal match
      - MARGIN_THRESHOLD: gap between top-2 scores → how dominant best result is

    Set INTERNAL_MIN_RERANK_SCORE in .env to override LOW_THRESHOLD at runtime.
    """

    if not contexts:
        return ConfidenceDecision(
            should_use_web=True,
            reason="No internal documents retrieved.",
            confidence="none",
        )

    rerank_scores = [
        ctx.rerank_score
        for ctx in contexts
        if ctx.rerank_score is not None
    ]

    if not rerank_scores:
        # Reranker did not run — fall back to fusion score heuristic
        top_fusion = max(ctx.fusion_score for ctx in contexts)
        if top_fusion < 0.05:
            return ConfidenceDecision(
                should_use_web=True,
                reason=f"No rerank scores available and top fusion score is weak ({top_fusion:.4f}).",
                confidence="low",
            )
        return ConfidenceDecision(
            should_use_web=False,
            reason=f"No rerank scores, but fusion score is acceptable ({top_fusion:.4f}).",
            confidence="medium",
        )

    rerank_scores.sort(reverse=True)
    top_score = rerank_scores[0]
    second_score = rerank_scores[1] if len(rerank_scores) > 1 else top_score - 999.0
    margin = top_score - second_score

    # Runtime-configurable floor (set INTERNAL_MIN_RERANK_SCORE in .env).
    # Default: 0.0 (logit boundary between negative and positive match).
    low_threshold: float = config.internal_min_rerank_score if config.internal_min_rerank_score is not None else 0.0
    high_threshold: float = 1.5    # logit above which we consider a strong match
    margin_threshold: float = 0.5  # how dominant the best result must be

    # No positive match at all
    if top_score < low_threshold:
        return ConfidenceDecision(
            should_use_web=True,
            reason=(
                f"Top rerank score ({top_score:.3f}) is below threshold ({low_threshold:.3f}). "
                "No strong internal match found."
            ),
            confidence="none" if top_score < 0 else "low",
        )

    # Weak match — retrieved something but not confidently relevant
    if top_score < high_threshold:
        return ConfidenceDecision(
            should_use_web=True,
            reason=(
                f"Top rerank score ({top_score:.3f}) is below high-confidence threshold "
                f"({high_threshold:.3f}). Augmenting with web."
            ),
            confidence="low",
        )

    # Strong top score but results are too clustered — ambiguous which doc is right
    if margin < margin_threshold:
        return ConfidenceDecision(
            should_use_web=False,
            reason=(
                f"Top score is strong ({top_score:.3f}) but margin over second result "
                f"is small ({margin:.3f}). Using internal results at medium confidence."
            ),
            confidence="medium",
        )

    # Clear winner — high top score with meaningful margin
    return ConfidenceDecision(
        should_use_web=False,
        reason=(
            f"Strong top rerank score ({top_score:.3f}) with clear margin "
            f"({margin:.3f}) over second result. High confidence in internal results."
        ),
        confidence="high",
    )



def filter_irrelevant_results(contexts: List[RetrievedContext], threshold_ratio: float = 0.6):
    if not contexts: return []
    
    top_score = contexts[0].rerank_score
    # If the top score is already low, the whole batch might be noise
    if top_score < 0: # BGE Reranker uses 0 as the neutral point
        return []

    # Only keep results that are at least 60% as good as the top result
    filtered = [
        ctx for ctx in contexts 
        if ctx.rerank_score > (top_score * threshold_ratio)
    ]
    return filtered