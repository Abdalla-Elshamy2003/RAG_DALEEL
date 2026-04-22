from __future__ import annotations


def faithfulness_metric(example, prediction, trace=None) -> float:
    answer = (getattr(prediction, "answer", "") or "").strip()
    context_block = (getattr(example, "context_block", "") or "").strip()
    if not answer:
        return 0.0
    if not context_block:
        return 1.0
    grounded_markers = ("[SOURCE ", "[DOCUMENT ")
    if any(marker in answer for marker in grounded_markers):
        return 1.0
    return 0.25
