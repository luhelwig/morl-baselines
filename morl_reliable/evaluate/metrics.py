import enum
from dataclasses import dataclass

import numpy as np

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_reliable.performance_indicators import hypervolume, maximum_utility_loss, expected_utility, cardinality, igd

# OBS: It needs to be default for all Agents and needs to add new metrics it is valuable.
# TODO: evaluate the effectiveness of the solution and evaluate the agent needs to receive the metric it can be custom or the ones bellow.
@dataclass(frozen=True)
class EvalContext:
    ref_point: np.ndarray
    weights: np.ndarray
    ref_front: list[np.ndarray] | None = None

class EvaluationMethod(enum.Enum):
    HV = "hv"
    EUM = "eum"
    CARD = "cardinality"
    MUL = "mul"
    IGD = "igd"

    @property
    def requires_ref_front(self) -> bool:
        return self in (EvaluationMethod.MUL, EvaluationMethod.IGD)

    def __call__(self, front: list[np.ndarray], ctx: EvalContext) -> float:
        if self.requires_ref_front and ctx.ref_front is None:
            raise ValueError(f"{self.name} requer ref_front")
        match self:
            # TODO: fix the args passed to the metrics methods;
            case EvaluationMethod.HV:
                return hypervolume(ctx.ref_point, front)
            case EvaluationMethod.EUM:
                return expected_utility(front, ctx.weights)
            case EvaluationMethod.CARD:
                return cardinality(front)
            case EvaluationMethod.MUL:
                return maximum_utility_loss(front, ctx.ref_front, ctx.weights)
            case EvaluationMethod.IGD:
                return igd(ctx.ref_front, front)

DEFAULT_METRICS: list[EvaluationMethod] = [
    EvaluationMethod.HV,
    EvaluationMethod.EUM,
    EvaluationMethod.CARD,
    EvaluationMethod.IGD,
    EvaluationMethod.MUL,
]