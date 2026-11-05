"""Registry of evaluation metrics for MORL Reliable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from morl_reliable.evaluate.context import EvalContext

MetricFunction = Callable[[np.ndarray, EvalContext], float]


@dataclass(frozen=True)
class Metric:
    """
    A named metric registered for use by the MORL Reliable ``FrontierEvaluator``.

    Instances are created by the ``register_metric`` decorator.

    Attributes:
        name: Unique name of the metric.
        fn: Function that computes the metric from a front and a context.
        requires: Names of the ``EvalContext`` fields that must be set
            (not None) for this metric to be computed.

    References:
        morl_reliable.evaluate.registry.register_metric
    """

    name: str
    fn: MetricFunction
    requires: frozenset[str] = frozenset()

    def __call__(self, front: np.ndarray, ctx: EvalContext) -> float:
        """
        Computes the metric for a Pareto front approximation.

        Args:
            front: Pareto front approximation to evaluate, as an array of shape
                (n_points, n_objectives). Each row is the expected vector return
                of one policy (objectives are maximized).
            ctx: Evaluation context providing the extra data the metric may need
                (reference point, weights, reference front, utility).

        Returns:
            The metric value for the given front.

        Raises:
            ValueError: If a field listed in ``requires`` is missing from ``ctx``.
        """
        if missing := ctx.missing(self.requires):
            raise ValueError(f"{self.name} requires {missing}")
        return self.fn(front, ctx)


_REGISTRY: dict[str, Metric] = {}


def register_metric(name: str, requires: Iterable[str] = ()):
    """
    Decorator that registers a metric function in the global registry.

    Args:
        name: Unique name of the metric.
        requires: Names of the ``EvalContext`` fields the metric needs.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If a metric with the same name is already registered.

    References:
        morl_reliable.evaluate.evaluator.FrontierEvaluator
        morl_reliable.evaluate.context.EvalContext

    Example:
        ```python
        @register_metric("my_metric", requires={"ref_point"})
        def my_metric(front, ctx):
            ...
        ```
    """

    def decorator(fn: MetricFunction) -> MetricFunction:
        if name in _REGISTRY:
            raise ValueError(f"Metric {name} already registered")

        _REGISTRY[name] = Metric(name, fn, frozenset(requires))
        return fn

    return decorator


def get_metric(metric: str | Metric) -> Metric:
    """
    Resolves a metric from the registry.

    Args:
        metric: Name of a registered metric, or a ``Metric`` object,
            which is returned as is.

    Returns:
        The resolved ``Metric``.

    Raises:
        ValueError: If no metric with the given name is registered.
    """
    if isinstance(metric, Metric):
        return metric

    try:
        return _REGISTRY[str(metric)]
    except KeyError:
        raise ValueError(f"Metric {metric} not registered") from None
