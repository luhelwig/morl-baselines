"""Registry-based evaluation of Pareto front approximations for MORL Reliable."""

from . import builtins as _builtins  # noqa: F401  (registra as métricas padrão)
from .builtins import DEFAULT_METRICS, MetricName
from .context import EvalContext
from .evaluator import FrontierEvaluator, OnMissingBehavior
from .registry import Metric, MetricFunction, get_metric, register_metric

__all__ = [
    "DEFAULT_METRICS",
    "EvalContext",
    "FrontierEvaluator",
    "Metric",
    "MetricFunction",
    "MetricName",
    "OnMissingBehavior",
    "get_metric",
    "register_metric",
]
