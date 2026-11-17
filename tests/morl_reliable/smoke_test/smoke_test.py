"""Smoke test for the post-training evaluation pipeline.

Every built-in ``Algorithm`` is trained for a few steps, saved, and then
evaluated by name, by class and through a loader. The loaded agent must give
the same front as the trained one.
"""

import importlib.util

import mo_gymnasium as mo_gym
import numpy as np
import pytest

from morl_baselines.common.weights import equally_spaced_weights, extrema_weights
from morl_reliable import (
    Algorithm,
    EvalParams,
    evaluate_algorithm,
    get_algorithm,
    make_loader,
)
from morl_reliable.evaluate import OnMissingBehavior
from morl_reliable.morl_reliable import run_agent


DISCRETE_ENV = "deep-sea-treasure-v0"
CONTINUOUS_ENV = "mo-mountaincarcontinuous-v0"

ALGO_KWARGS = {"net_arch": [32, 32], "buffer_size": 1_000, "batch_size": 32, "learning_starts": 50, "gradient_updates": 1}


def _train(agent, env):
    agent.train(total_timesteps=200)


def _train_gpi_iteration(agent, env):
    weight_support = extrema_weights(2)
    agent.train_iteration(total_timesteps=200, weight=weight_support[0], weight_support=weight_support, eval_env=env)


def _train_capql(agent, env):
    agent.train(total_timesteps=200, eval_env=env, ref_point=np.array([-1000.0, -1000.0]), eval_freq=10_000)


CASES = {
    Algorithm.ENVELOPE: (DISCRETE_ENV, _train),
    Algorithm.GPI_PD_DISCRETE: (DISCRETE_ENV, _train_gpi_iteration),
    Algorithm.GPI_LS_DISCRETE: (DISCRETE_ENV, _train_gpi_iteration),
    Algorithm.GPI_LS_DISCRETE_JAX: (DISCRETE_ENV, _train_gpi_iteration),
    Algorithm.GPI_PD_CONTINUOUS: (CONTINUOUS_ENV, _train_gpi_iteration),
    Algorithm.GPI_LS_CONTINUOUS: (CONTINUOUS_ENV, _train_gpi_iteration),
    Algorithm.GPI_LS_CONTINUOUS_JAX: (CONTINUOUS_ENV, _train_gpi_iteration),
    Algorithm.CAPQL: (CONTINUOUS_ENV, _train_capql),
}

NEEDS_JAX = {Algorithm.GPI_LS_DISCRETE_JAX, Algorithm.GPI_LS_CONTINUOUS_JAX}
HAS_JAX = importlib.util.find_spec("jax") is not None
ALGORITHMS = [
    pytest.param(a, marks=pytest.mark.skipif(a in NEEDS_JAX and not HAS_JAX, reason="needs the jax extra")) for a in Algorithm
]


def print_metrics(results):
    for name, r in results.items():
        print(f"\n{name}")
        for metric, value in r.metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"  front ({len(r.front)} points):\n{r.front}")


def test_every_algorithm_has_a_case():
    assert set(CASES) == set(Algorithm)


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=str)
def test_evaluate_algorithm(algorithm, tmp_path):
    env_id, train = CASES[algorithm]
    env = mo_gym.make(env_id)

    agent = get_algorithm(algorithm)(env, log=False, seed=0, **ALGO_KWARGS)
    train(agent, env)
    agent.save(save_replay_buffer=False, save_dir=str(tmp_path), filename="seed_0")

    ref_front = env.unwrapped.pareto_front(gamma=0.99) if hasattr(env.unwrapped, "pareto_front") else None
    params = EvalParams(
        env_id=env_id,
        ref_point=[0.0, -25.0] if env_id == DISCRETE_ENV else [-1000.0, -1000.0],
        weights=equally_spaced_weights(2, n=5),
        ref_front=ref_front,
        gamma=0.99,
        num_episodes=1,
        on_missing=OnMissingBehavior.RAISE if ref_front is not None else OnMissingBehavior.WARN,
    )
    expected_front = run_agent(agent, mo_gym.make(env_id), params)

    by_name = evaluate_algorithm(algorithm, tmp_path, params, algo_kwargs=ALGO_KWARGS)
    by_class = evaluate_algorithm(get_algorithm(algorithm), tmp_path, params, algo_kwargs=ALGO_KWARGS)
    by_loader = evaluate_algorithm(make_loader(algorithm, ALGO_KWARGS), tmp_path, params)
    print_metrics(by_name)
    env.close()

    for results in (by_name, by_class, by_loader):
        assert set(results) == {"seed_0"}
        np.testing.assert_allclose(results["seed_0"].front, expected_front, rtol=1e-5)
        assert "Hypervolume" in results["seed_0"].metrics
