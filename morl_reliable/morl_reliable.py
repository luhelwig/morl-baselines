"""Post-training evaluation of MORL agents.

For each saved checkpoint, the greedy policy is rolled out for a fixed set of
preference weights. This yields an approximation of the Pareto front, which is
then scored with the MORL Reliable metrics.

Pipeline:
    evaluate_algorithm -> loader -> evaluate_agent -> run_agent -> FrontierEvaluator

Example:
    ```python
    params = EvalParams(
        env_id="deep-sea-treasure-v0",
        ref_point=[0.0, -25.0],
        weights=equally_spaced_weights(2, n=100),
    )
    loader = class_loader(Envelope, net_arch=[256, 256, 256, 256], load_kwargs={"load_replay_buffer": False})
    results = evaluate_algorithm(Path("weights"), params, loader)
    results["seed_0"].metrics
    # {"Hypervolume": ..., "Expected Utility": ..., ...}
    ```
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.common.pareto import filter_pareto_dominated
from morl_reliable import Loader
from morl_reliable.evaluate import (
    DEFAULT_METRICS,
    EvalContext,
    FrontierEvaluator,
    OnMissingBehavior,
)
from morl_reliable.evaluate.registry import Metric
from morl_reliable.evaluation import policy_evaluation_mo


@dataclass
class EvalParams:
    """Evaluation protocol shared by every agent being compared.

    All fields must be identical across algorithms and seeds; otherwise the
    resulting metrics are not comparable.

    Attributes:
        env_id: MO-Gymnasium id of the environment the agents were trained on.
        ref_point: Reference point for the hypervolume, shape (n_objectives,).
            Must be dominated by every point of interest.
        weights: Preference weight vectors, shape (n_weights, n_objectives).
            Each one yields one point of the front; they are also used by
            the utility-based metrics.
        ref_front: Known Pareto front, shape (n_points, n_objectives), if any.
            Required by metrics such as IGD and MUL. Must use the same kind
            of return (discounted or not) as ``discounted``.
        gamma: Discount factor for the discounted returns. If None, each agent's
            own ``gamma`` is used, which makes fronts of agents trained with
            different discounts not comparable.
        num_episodes: Number of episodes averaged for each weight vector.
        discounted: If True, the front is built from discounted vector returns.
        seed: Seed for the evaluation environment.
        metrics: Metrics to compute, by registered name or as ``Metric`` objects.
            Defaults to ``DEFAULT_METRICS``.
        on_missing: What to do when a metric's requirements are missing from
            the evaluation context.

    Raises:
        ValueError: If ``ref_point`` and ``weights`` have different numbers
            of objectives.
    """

    env_id: str
    ref_point: np.ndarray
    weights: np.ndarray

    ref_front: np.ndarray | None = None
    gamma: float | None = None
    num_episodes: int = 5
    discounted: bool = True
    seed: int = 0
    metrics: list[str | Metric] = field(default_factory=lambda: list(DEFAULT_METRICS))
    on_missing: OnMissingBehavior = OnMissingBehavior.RAISE

    def __post_init__(self):
        """Converts array-like fields to ``float32`` arrays and validates their shapes."""
        self.ref_point = np.asarray(self.ref_point, dtype=np.float32)
        self.weights = np.atleast_2d(np.asarray(self.weights, dtype=np.float32))
        if self.ref_front is not None:
            self.ref_front = np.atleast_2d(np.asarray(self.ref_front, dtype=np.float32))

        if self.ref_point.shape[0] != self.weights.shape[1]:
            raise ValueError("ref_point and weights have different numbers of objectives")


class AgentEval(NamedTuple):
    """Evaluation result of a single agent.

    Attributes:
        front: Non-dominated points found by the agent,
            shape (n_points, n_objectives).
        metrics: Mapping from metric name to its value.
    """

    front: np.ndarray
    metrics: dict[str, float]


def run_agent(agent: Any, env: gym.Env, params: EvalParams) -> np.ndarray:
    """Builds the agent's Pareto front approximation.

    For each weight vector, the greedy policy is run for ``num_episodes``
    episodes and its average vector return becomes one point. Dominated
    points are then removed. The environment is reset with ``params.seed``
    first, so every agent sees the same sequence of episodes.

    Args:
        agent: Trained agent.
        env: Evaluation environment.
        params: Evaluation protocol.

    Returns:
        Non-dominated points, shape (n_points, n_objectives).
    """
    env.reset(seed=params.seed)

    idx = 3 if params.discounted else 2

    points = [policy_evaluation_mo(agent, env, w=w, rep=params.num_episodes, gamma=params.gamma)[idx] for w in params.weights]
    return np.asarray(filter_pareto_dominated(points))


def make_evaluator(params: EvalParams) -> FrontierEvaluator:
    """Creates the metric evaluator shared by all agents.

    Args:
        params: Evaluation protocol.

    Returns:
        A ``FrontierEvaluator`` configured with ``params``.

    Raises:
        ValueError: If a metric's requirements are missing and
            ``params.on_missing`` is ``RAISE``.
    """
    ctx = EvalContext(ref_point=params.ref_point, weights=params.weights, ref_front=params.ref_front)
    return FrontierEvaluator(ctx=ctx, metrics=params.metrics, on_missing=params.on_missing)


def evaluate_agent(agent: Any, env: gym.Env, evaluator: FrontierEvaluator, params: EvalParams) -> AgentEval:
    """Builds the agent's front and computes its metrics.

    Args:
        agent: Trained agent.
        env: Evaluation environment.
        evaluator: Metric evaluator, see ``make_evaluator``.
        params: Evaluation protocol.

    Returns:
        The agent's front and its metrics.
    """
    front = run_agent(agent, env, params)
    return AgentEval(front, evaluator(front))


def evaluate_algorithm(
    agents_dir: Path,
    params: EvalParams,
    loader: Loader,
    pattern: str = "*.tar",
) -> dict[str, AgentEval]:
    """Evaluates every checkpoint of an algorithm (typically one per seed).

    Args:
        agents_dir: Directory containing the checkpoints.
        params: Evaluation protocol, applied identically to every agent.
        loader: Function that rebuilds an agent from a checkpoint path and
            the environment. See ``morl_reliable.loader``.
        pattern: Glob pattern used to find the checkpoints in ``agents_dir``.

    Returns:
        Mapping from checkpoint file name (without extension) to its result.

    Raises:
        FileNotFoundError: If no file in ``agents_dir`` matches ``pattern``.
    """
    files = sorted(agents_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} checkpoints found in {agents_dir}")

    env = mo_gym.make(params.env_id)
    evaluator = make_evaluator(params)

    results: dict[str, AgentEval] = {}
    try:
        for file in files:
            agent = loader(file, env)
            results[file.stem] = evaluate_agent(agent, env, evaluator, params)
    finally:
        env.close()
    return results
