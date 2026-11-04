# TODO: Classes pydoc
import warnings
from enum import StrEnum
from typing import Sequence

import numpy as np

from morl_reliable.evaluate.builtins import REQUIRED_METRICS
from morl_reliable.evaluate.context import EvalContext
from morl_reliable.evaluate.registry import get_metric, Metric


class OnMissingBehavior(StrEnum):
    RAISE = "raise"
    WARN = "warn"

class Evaluator:
    ctx: EvalContext
    metrics: dict[str, Metric]

    def __init__(
            self,
            ctx: EvalContext,
            extra_metrics: Sequence[str | Metric] = (),
            on_missing: OnMissingBehavior = OnMissingBehavior.RAISE
    ):
        self.ctx = ctx
        self.metrics = {}

        for m in map(get_metric, REQUIRED_METRICS):
            if missing := ctx.missing(m.requires):
                raise ValueError(f"Required metric {m.name} requires {missing}")
            self.metrics[m.name] = m
        for m in map(get_metric, extra_metrics):
            if self._already_included(m):
                continue

            missing = ctx.missing(m.requires)
            if not missing:
                self.metrics[m.name] = m
                continue

            if on_missing == OnMissingBehavior.RAISE:
                raise ValueError(f"{m.name} requires {missing}")
            if on_missing == OnMissingBehavior.WARN:
                warnings.warn(f"Skipping {m.name}: {missing} is missing")

    def _already_included(self, metric: Metric) -> bool:
        existing = self.metrics.get(metric.name)
        if existing is None:
            return False
        if existing is metric:
            return True

        raise ValueError(f"metric '{metric.name}' already defined")

    def __call__(self, front) -> dict[str, float]:
        front = np.atleast_2d(np.asarray(front, dtype=np.float32))
        return {name: m(front, self.ctx) for name, m in self.metrics.items()}

    def __str__(self) -> str:
        return str(self.metrics)

    def __repr__(self) -> str:
        return repr(self.metrics)

