"""Evaluation context holding the data that metrics may require."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EvalContext:
    """Data shared by all metrics in an evaluation.

    Each metric declares which of these fields it needs (see
    ``register_metric``); fields left as ``None`` are considered missing.

    Attributes:
        ref_point: Reference point for the hypervolume, of shape
            (n_objectives,). It must be dominated by every point of the
            front (objectives are maximized).
        weights: Weight vectors used by utility-based metrics (EUM, MUL),
            of shape (n_weights, n_objectives).
        ref_front: Known (true) Pareto front of the problem, of shape
            (n_points, n_objectives), used by IGD and MUL.
        utility: Utility function called as ``utility(weights, point)``
            and returning a float. Defaults to ``np.dot`` (linear utility).

    References:
        morl_reliable.evaluate.registry.register_metric
        morl_reliable.evaluate.evaluator.FrontierEvaluator

    Example:
        ```python
        ctx = EvalContext(
            ref_point=np.array([0.0, 0.0]),
            weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            ref_front=np.array([[10.0, 1.0], [6.0, 5.0], [1.0, 9.0]]),
        )
        ```
    """

    ref_point: np.ndarray | None = None
    weights: np.ndarray | None = None
    ref_front: np.ndarray | None = None
    utility: Callable = np.dot

    def missing(self, requires: frozenset[str]) -> set[str]:
        """Returns the required fields that are not available in this context.

        Args:
            requires: Names of the fields a metric needs.

        Returns:
            The names in ``requires`` whose value is ``None``. Names that
            are not fields of ``EvalContext`` are also reported as missing.
        """
        return {r for r in requires if getattr(self, r, None) is None}
