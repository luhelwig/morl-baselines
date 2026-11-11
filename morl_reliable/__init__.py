"""MORL Reliable: reliable evaluation of multi-objective RL agents."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from morl_reliable.morl_reliable import AgentEval, EvalParams, evaluate_algorithm

__all__ = [
    "AgentEval",
    "EvalParams",
    "evaluate_algorithm",
]


def __getattr__(name: str):
    """Lazily imports the evaluator to avoid a circular import with morl_baselines."""
    if name in __all__:
        from morl_reliable import morl_reliable

        return getattr(morl_reliable, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")