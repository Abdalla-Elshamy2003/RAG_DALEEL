from __future__ import annotations

import dspy


class GenerateRAGAnswerSignature(dspy.Signature):
    """Answer the user using only the supplied evidence.
    Every factual claim must be grounded in the evidence and cited with the
    requested citation style. If evidence is insufficient, say exactly:
    "I don't have sufficient information in the available sources to answer this question."
    Make the answer meaningfully detailed, not brief.
    Start with a direct answer, then add a rich Details section that explains
    the main points, supporting evidence, important nuances, and practical
    interpretation when useful. Add Notes only when necessary.
    Prefer complete, professional answers over short summaries.
    The writing should feel natural, human, and polished, not robotic or template-like."""

    question = dspy.InputField(desc="Original user question.")
    rewritten_question = dspy.InputField(desc="Polished final answering query.")
    answer_style = dspy.InputField(desc="Requested answer style.")
    source_scope = dspy.InputField(desc="Scope of evidence, such as internal KB or web.")
    language_instruction = dspy.InputField(desc="Language policy for the final answer.")
    citation_style = dspy.InputField(
        desc="Required citation format, such as [SOURCE n] or [DOCUMENT n]."
    )
    context_block = dspy.InputField(desc="All evidence provided to the model.")
    answer = dspy.OutputField(
        desc="Final grounded answer with citations, written in a rich, detailed, natural, and human-sounding style."
    )


class GenerateGeneralAnswerSignature(dspy.Signature):
    """Answer the user directly and professionally without citing documents.
    Be accurate, clear, and explicit about uncertainty. Keep the answer in the
    requested language and structure it with a direct answer followed by a
    substantial Details section. The answer should be comfortably detailed,
    explanatory, and useful, not just a short reply. Add Notes only when useful.
    The writing should sound natural and human, not robotic."""

    question = dspy.InputField(desc="Original user question.")
    rewritten_question = dspy.InputField(desc="Polished final answering query.")
    answer_style = dspy.InputField(desc="Requested answer style.")
    language_instruction = dspy.InputField(desc="Language policy for the final answer.")
    answer = dspy.OutputField(
        desc="Final professional answer with stronger detail, explanation, depth, and a natural human tone."
    )
