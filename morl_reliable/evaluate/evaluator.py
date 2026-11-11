"""Frontier Evaluator that computes MORL Reliable metrics over a Pareto front."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from morl_reliable.evaluate.builtins import DEFAULT_METRICS
from morl_reliable.evaluate.context import EvalContext
from morl_reliable.evaluate.registry import Metric, get_metric


class OnMissingBehavior(StrEnum):
    """How ``FrontierEvaluator`` handles metrics with missing requirements.

    Attributes:
        RAISE: Raise a ``ValueError``.
        WARN: Skip the metric and emit a warning.
    """

    RAISE = "raise"
    WARN = "warn"


class FrontierEvaluator:
    """Computes the MORL Reliable metrics for a Pareto front approximation.

    By default, the metrics in ``DEFAULT_METRICS`` are computed. Custom
    metrics can be registered with ``register_metric`` and passed in
    ``metrics``.

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
        ev = FrontierEvaluator(ctx)
        ev(np.array([[9.0, 1.0], [5.0, 5.0]]))
        # {"Hypervolume": ..., "Expected Utility": ..., ...}
        ```
    """

    ctx: EvalContext
    metrics: dict[str, Metric]

    def __init__(
        self,
        ctx: EvalContext,
        metrics: Sequence[str | Metric] = DEFAULT_METRICS,
        on_missing: OnMissingBehavior = OnMissingBehavior.RAISE,
    ):
        """Builds the set of metrics to evaluate and validates it against ``ctx``.

        Passing ``metrics`` replaces ``DEFAULT_METRICS``. With the defaults,
        ``ctx`` must provide ``ref_point``, ``ref_front`` and ``weights``. If a
        metric's requirements are missing from ``ctx``, ``on_missing`` decides
        whether to raise or to skip the metric with a warning. Passing the same
        ``Metric`` more than once has no effect.

        Args:
            ctx: Evaluation context shared by all metrics.
            metrics: Metrics to compute, by registered name or as ``Metric``
                objects. Defaults to ``DEFAULT_METRICS``.
            on_missing: What to do when a metric's requirements are not met
                by ``ctx``.

        Raises:
            ValueError: If a metric's requirements are missing from ``ctx`` and
                ``on_missing`` is ``RAISE``, if a metric name is not registered,
                or if two different metrics share the same name.
        """
        self.ctx = ctx
        self.metrics = {}

        for m in map(get_metric, metrics):
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
        """Checks whether ``metric`` is already in ``self.metrics``.

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
        """Computes every configured metric for a Pareto front approximation.

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
        """Returns the configured metrics as a readable string."""
        return str(self.metrics)

    def __repr__(self) -> str:
        """Returns the representation of the configured metrics."""
        return repr(self.metrics)
