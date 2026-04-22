from __future__ import annotations

import dspy


class RouteRetrievalSignature(dspy.Signature):
    """Decide whether the answering stage should use evidence-grounded RAG or
    a general-answer fallback. Prefer RAG whenever usable evidence exists."""

    query = dspy.InputField(desc="Original user question.")
    has_context = dspy.InputField(desc="Whether evidence context is available.")
    confidence = dspy.InputField(desc="Confidence label such as high, medium, low, or none.")
    used_web = dspy.InputField(desc="Whether web sources were used in retrieval.")
    route = dspy.OutputField(desc="Either 'rag' or 'general'.")
    rationale = dspy.OutputField(desc="Short reason for the routing decision.")
