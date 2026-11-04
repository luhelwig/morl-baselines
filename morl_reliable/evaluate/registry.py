# TODO: Methods and Classes pydoc
from dataclasses import dataclass
from typing import Callable

import numpy as np

from morl_reliable.evaluate.context import EvalContext

MetricFunction = Callable[[np.ndarray, EvalContext], float]

@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFunction
    requires: frozenset[str] = frozenset()

    def __call__(self, front, ctx: EvalContext) -> float:
        if missing := ctx.missing(self.requires):
            raise ValueError(f"{self.name} requires {missing}")
        return self.fn(front, ctx)

_REGISTRY: dict[str, Metric] = {}

def register_metric(name: str, requires=()):
    def decorator(fn: MetricFunction) -> MetricFunction:
        if name in _REGISTRY:
            raise ValueError(f"Metric {name} already registered")

        _REGISTRY[name] = Metric(name, fn, requires)
        return fn
    return decorator

def get_metric(metric: str | Metric) -> Metric:
    if isinstance(metric, Metric):
        return metric

    try :
        return _REGISTRY[str(metric)]
    except KeyError:
        raise ValueError(f"Metric {metric} not registered")
