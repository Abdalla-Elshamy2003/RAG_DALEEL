from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import dspy

from ..metrics.answer_metric import answer_metric
from ..settings import SAVED_PROGRAMS_DIR
from .bootstrap_fewshot import build_bootstrap_fewshot


def _save_compilation_metadata(
    *,
    path: Path,
    program_name: str,
    trainset_size: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "compiled": True,
                "program_name": program_name,
                "trainset_size": trainset_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def compile_rag_program(
    *,
    student,
    trainset: Iterable[dspy.Example],
    teacher=None,
    metric=answer_metric,
    output_path: Optional[Path] = None,
):
    trainset = list(trainset)
    teleprompter = build_bootstrap_fewshot(metric=metric)
    compiled = teleprompter.compile(student=student, teacher=teacher, trainset=trainset)
    metadata_path = output_path or (SAVED_PROGRAMS_DIR / "compiled_rag.json")
    _save_compilation_metadata(
        path=metadata_path,
        program_name="rag",
        trainset_size=len(trainset),
    )
    return compiled


def compile_router_program(
    *,
    student,
    trainset: Iterable[dspy.Example],
    teacher=None,
    metric=answer_metric,
    output_path: Optional[Path] = None,
):
    trainset = list(trainset)
    teleprompter = build_bootstrap_fewshot(metric=metric)
    compiled = teleprompter.compile(student=student, teacher=teacher, trainset=trainset)
    metadata_path = output_path or (SAVED_PROGRAMS_DIR / "compiled_router.json")
    _save_compilation_metadata(
        path=metadata_path,
        program_name="router",
        trainset_size=len(trainset),
    )
    return compiled
