"""Evaluator that computes MORL Reliable metrics over a Pareto front."""
import warnings
from enum import StrEnum
from typing import Sequence

import numpy as np

from morl_reliable.evaluate.builtins import REQUIRED_METRICS
from morl_reliable.evaluate.context import EvalContext
from morl_reliable.evaluate.registry import get_metric, Metric


class OnMissingBehavior(StrEnum):
    """
    What ``Evaluator`` does when an extra metric's requirements are
    missing from the context.

    Attributes:
        RAISE: Raise a ``ValueError``.
        WARN: Skip the metric and emit a warning.
    """
    RAISE = "raise"
    WARN = "warn"

class Evaluator:
    """
    Computes the MORL Reliable metrics for a Pareto front approximation.

    The metrics in ``REQUIRED_METRICS`` are always computed. Custom metrics
    can be registered with ``register_metric`` and passed in
    ``extra_metrics``.

    Attributes:
        ctx: Evaluation context shared by all metrics.
        metrics: Mapping from metric name to the ``Metric`` that will be
            computed.

    References:
        morl_reliable.evaluate.registry.register_metric

        morl_reliable.evaluate.context.EvalContext

    Example:
        ```python
        ctx = EvalContext(
            ref_point=np.array([0.0, 0.0]),
            weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            ref_front=np.array([[10.0, 1.0], [6.0, 5.0], [1.0, 9.0]]),
        )
        ev = Evaluator(ctx)
        ev(np.array([[9.0, 1.0], [5.0, 5.0]]))
        # {"Hypervolume": ..., "Expected Utility": ..., ...}
        ```
    """
    ctx: EvalContext
    metrics: dict[str, Metric]

    def __init__(
            self,
            ctx: EvalContext,
            extra_metrics: Sequence[str | Metric] = (),
            on_missing: OnMissingBehavior = OnMissingBehavior.RAISE
    ):
        """
        Builds the set of metrics to evaluate and validates it against ``ctx``.

        All metrics in ``REQUIRED_METRICS`` are always included, so ``ctx`` must
        provide every field they need (currently ``ref_point``, ``ref_front``
        and ``weights``). Extra metrics are added after them; if one of their
        requirements is missing, ``on_missing`` decides whether to raise or to
        skip the metric with a warning. Passing the same ``Metric`` more than
        once has no effect.

        Args:
            ctx: Evaluation context shared by all metrics.
            extra_metrics: Additional metrics to compute, by registered name or as
                ``Metric`` objects.
            on_missing: What to do when an extra metric's requirements are not met
                by ``ctx``. Required metrics always raise.

        Raises:
            ValueError: If a required metric's requirements are missing from ``ctx``,
                if an extra metric's requirements are missing and ``on_missing`` is
                ``RAISE``, if a metric name is not registered, or if two
                different metrics share the same name.
        """
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
        """
        Checks whether ``metric`` is already in ``self.metrics``.

        Args:
            metric: Metric to check.

        Returns:
            True if this same ``Metric`` object is already included,
            False if no metric with its name is included.

        Raises:
            ValueError: If a different metric with the same name is
                already included.
        """
        existing = self.metrics.get(metric.name)
        if existing is None:
            return False
        if existing is metric:
            return True

        raise ValueError(f"metric '{metric.name}' already defined")

    def __call__(self, front: np.ndarray) -> dict[str, float]:
        """
        Computes every configured metric for a Pareto front approximation.

        Args:
            front: Pareto front approximation, array-like of shape
                (n_points, n_objectives). It is converted to a 2D
                ``float32`` array before evaluation.

        Returns:
            A mapping from metric name to its value.
        """
        front = np.atleast_2d(np.asarray(front, dtype=np.float32))
        return {name: m(front, self.ctx) for name, m in self.metrics.items()}

    def __str__(self) -> str:
        return str(self.metrics)

    def __repr__(self) -> str:
        return repr(self.metrics)