from __future__ import annotations

import re


_CITATION_RE = re.compile(r"\[(SOURCE|DOCUMENT)\s+\d+\]")


def citation_metric(example, prediction, trace=None) -> float:
    answer = getattr(prediction, "answer", "") or ""
    required = getattr(example, "requires_citations", True)
    if not required:
        return 1.0
    citations = _CITATION_RE.findall(answer)
    return 1.0 if citations else 0.0
