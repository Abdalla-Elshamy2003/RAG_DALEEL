from __future__ import annotations

import dspy


class JudgeAnswerQualitySignature(dspy.Signature):
    """Judge whether a predicted answer sufficiently matches a reference answer
    while preserving important meaning, coverage, and professional quality."""

    question = dspy.InputField(desc="Original user question.")
    predicted_answer = dspy.InputField(desc="Model-generated answer.")
    reference_answer = dspy.InputField(desc="Expected reference answer.")
    quality_score = dspy.OutputField(desc="A score from 0.0 to 1.0.")
    rationale = dspy.OutputField(desc="Short justification for the score.")


class JudgeFaithfulnessSignature(dspy.Signature):
    """Judge whether an answer is faithful to the supplied evidence and whether
    its citations align with the retrieved context."""

    question = dspy.InputField(desc="Original user question.")
    predicted_answer = dspy.InputField(desc="Model-generated answer.")
    context_block = dspy.InputField(desc="Evidence shown to the model.")
    faithfulness_score = dspy.OutputField(desc="A score from 0.0 to 1.0.")
    rationale = dspy.OutputField(desc="Short justification for the score.")
