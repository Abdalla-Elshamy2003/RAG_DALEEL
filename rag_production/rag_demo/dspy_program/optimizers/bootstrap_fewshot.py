from __future__ import annotations

from typing import Callable, Optional

import dspy


def build_bootstrap_fewshot(
    *,
    metric: Optional[Callable] = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
    max_rounds: int = 1,
) -> dspy.BootstrapFewShot:
    return dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
    )
