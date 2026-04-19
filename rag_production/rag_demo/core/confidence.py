
from __future__ import annotations

from typing import List

from .config import RAGConfig
from .schemas import ConfidenceDecision, RetrievedContext

import numpy as np

def evaluate_internal_results(
    *,
    contexts: List[RetrievedContext],
    config: RAGConfig,
) -> ConfidenceDecision:
    """
    Evaluate if internal retrieved documents are sufficient or if we should fallback to web search.
    Uses a more dynamic, feature-based approach instead of hard-coded rules.
    """

    if not contexts:
        return ConfidenceDecision(
            should_use_web=True,
            reason="No internal documents were retrieved.",
            confidence="none",
        )

    # Feature extraction from contexts
    rerank_scores = [ctx.rerank_score for ctx in contexts if ctx.rerank_score is not None]
    fusion_scores = [ctx.fusion_score for ctx in contexts]
    num_contexts = len(contexts)
    
    # Dynamic thresholds based on historical data or calculated percentiles
    score_threshold = np.percentile(rerank_scores, 80) if rerank_scores else 0.0  # Use 80th percentile as threshold
    context_score_mean = np.mean(fusion_scores) if fusion_scores else 0.0
    context_score_std = np.std(fusion_scores) if fusion_scores else 0.0

    # Confidence Decision Logic:
    # More robust decisions based on the percentile and distribution of scores.
    
    if np.all(np.array(rerank_scores) < score_threshold):
        # If the rerank scores are too low, we should consider the web
        return ConfidenceDecision(
            should_use_web=True,
            reason=f"Rerank scores are too low (below {score_threshold:.4f} percentile).",
            confidence="low",
        )
    
    if num_contexts < 3:
        # If fewer than 3 contexts, confidence drops
        return ConfidenceDecision(
            should_use_web=True,
            reason="Insufficient number of contexts retrieved (fewer than 3).",
            confidence="medium",
        )

    # If we have a high mean fusion score and most scores are above the threshold, we have high confidence
    if context_score_mean > score_threshold and context_score_std < (context_score_mean / 2):
        return ConfidenceDecision(
            should_use_web=False,
            reason="Internal results are confident with high consistency.",
            confidence="high",
        )

    # For scenarios in between, make a dynamic decision based on score variation and the number of contexts
    return ConfidenceDecision(
        should_use_web=False,
        reason="Internal results are available, but confidence is medium due to score variability.",
        confidence="medium",
    )