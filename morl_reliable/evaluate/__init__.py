from .context import EvalContext
from .registry import Metric, MetricFunction, register_metric, get_metric
from . import builtins as _builtins  # registra as métricas padrão
from .builtins import MetricName, REQUIRED_METRICS
from .evaluator import Evaluator, OnMissingBehavior

__all__ = [
    "Evaluator",
    "EvalContext",
    "OnMissingBehavior",
    "Metric",
    "MetricFunction",
    "MetricName",
    "REQUIRED_METRICS",
    "register_metric",
    "get_metric",
]