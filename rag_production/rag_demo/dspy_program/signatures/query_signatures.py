from __future__ import annotations

import dspy


class AnalyzeQuerySignature(dspy.Signature):
    """Analyze the user's question for a professional Arabic/English RAG workflow.
    Produce a concise intent summary, a retrieval-focused rewrite, and explicit
    language guidance without inventing facts outside the question.
    If the user's question is in Arabic, the downstream answer must be in Arabic,
    not English, except for unavoidable technical terms."""

    query = dspy.InputField(desc="Original user question.")
    answer_style = dspy.InputField(desc="Desired answer style guidance.")
    intent = dspy.OutputField(desc="Short description of user intent.")
    retrieval_query = dspy.OutputField(
        desc="A clean retrieval-focused rewrite of the question."
    )
    language_instruction = dspy.OutputField(
        desc="One sentence telling the downstream module which language to answer in, with strict Arabic-only guidance for Arabic questions."
    )


class RewriteQuerySignature(dspy.Signature):
    """Rewrite the user question into a polished final answering query.
    Keep meaning unchanged, preserve requested language, and optimize clarity."""

    query = dspy.InputField(desc="Original user question.")
    intent = dspy.InputField(desc="Intent summary of the question.")
    answer_style = dspy.InputField(desc="Desired answer style guidance.")
    rewritten_query = dspy.OutputField(
        desc="Polished final question for answer generation."
    )
