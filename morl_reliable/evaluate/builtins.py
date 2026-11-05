"""Built-in metrics computed by every MORL Reliable ``Evaluator``."""
import enum

from morl_reliable.evaluate.context import EvalContext
from morl_reliable.evaluate.registry import register_metric
from morl_reliable.performance import (
    hypervolume,
    igd,
    sparsity,
    expected_utility,
    cardinality,
    maximum_utility_loss,
)


class MetricName(enum.StrEnum):
    """
    Short aliases for the names of the built-in metrics.

    Each member's value is the name the metric is registered under, which
    is also its key in the ``Evaluator`` output. Since this is a
    ``StrEnum``, a member can be used anywhere a metric name is expected.

    Example:
        ```python
        results = ev(front)
        results[MetricName.HV]  # same as results["Hypervolume"]
        ```
    """
    HV       = "Hypervolume"
    IGD      = "Inverted Generation Distance"
    SPARSITY = "Sparsity"
    EUM      = "Expected Utility"
    CARD     = "Cardinality"
    MUL      = "Maximum Utility Loss"

REQUIRED_METRICS = list(MetricName)
"""Metrics that every ``Evaluator`` always computes."""

@register_metric(MetricName.HV, requires={"ref_point"})
def _hv_metric(front, ctx: EvalContext):
    """Hypervolume of ``front`` with respect to ``ctx.ref_point``."""
    assert ctx.ref_point is not None
    return hypervolume(ref_point=ctx.ref_point, points=front)

@register_metric(MetricName.IGD, requires={"ref_front"})
def _inverted_generation_distance_metric(front, ctx: EvalContext):
    """Inverted Generational Distance from ``front`` to ``ctx.ref_front``."""
    assert ctx.ref_front is not None
    return igd(known_front=ctx.ref_front,current_estimate=front)

@register_metric(MetricName.SPARSITY, requires={})
def _sm_metric(front, ctx: EvalContext):
    """Sparsity of ``front``; does not use the context."""
    assert ctx.ref_point is not None
    return sparsity(front=front)

@register_metric(MetricName.EUM, requires={"weights"})
def _eum_metric(front, ctx: EvalContext):
    """Expected utility of ``front`` over ``ctx.weights``."""
    assert ctx.weights is not None
    return expected_utility(front, weights_set=ctx.weights, utility=ctx.utility)

@register_metric(MetricName.CARD, requires={"weights"})
def _cardinality_metric(front, ctx: EvalContext):
    """Cardinality of ``front``; does not use the context."""
    return cardinality(front)

@register_metric(MetricName.MUL, requires={"ref_front", "weights"})
def _maximum_utility_loss_metric(front, ctx: EvalContext):
    """Maximum utility loss of ``front`` relative to ``ctx.ref_front``."""
    assert ctx.ref_front is not None
    assert ctx.weights is not None
    return maximum_utility_loss(front, reference_set=ctx.ref_front, weights_set=ctx.weights, utility=ctx.utility)