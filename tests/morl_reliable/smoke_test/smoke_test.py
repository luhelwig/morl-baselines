"""Smoke test for the post-training evaluation pipeline."""

from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.envelope.envelope import Envelope
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_reliable import EvalParams, class_loader, evaluate_algorithm
from morl_reliable.evaluate import OnMissingBehavior

def print_metrics(results):
    for name, r in results.items():
        print(f"\n{name}")
        for metric, value in r.metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"  front ({len(r.front)} points):\n{r.front}")

def test_evaluate_algorithm_envelope(tmp_path):
    env_id = "deep-sea-treasure-v0"
    net_arch = [32, 32]

    agent = Envelope(mo_gym.make(env_id), net_arch=net_arch, buffer_size=1_000, learning_starts=50, log=False, seed=0)
    agent.train(total_timesteps=300)
    agent.save(save_replay_buffer=False, save_dir=str(tmp_path), filename="seed_0")

    ref_env = mo_gym.make(env_id)
    params = EvalParams(
        env_id=env_id,
        ref_point=[0.0, -25.0],
        weights=equally_spaced_weights(2, n=5),
        ref_front=ref_env.unwrapped.pareto_front(gamma=0.99),
        gamma=0.99,
        num_episodes=1,
        on_missing=OnMissingBehavior.RAISE,
    )
    loader = class_loader(Envelope, net_arch=net_arch, buffer_size=1_000, load_kwargs={"load_replay_buffer": False})

    results = evaluate_algorithm(tmp_path, params, loader)
    print_metrics(results)

    assert set(results) == {"seed_0"}
    assert results["seed_0"].front.shape[1] == 2
    assert results["seed_0"].metrics

def test_evaluate_algorithm_gpi_pd(tmp_path):
    env_id = "deep-sea-treasure-v0"
    net_arch = [32, 32]
    ref_point = [0.0, -25.0]
    env = mo_gym.make(env_id)
    agent = GPIPD(env, net_arch=net_arch, buffer_size=1_000, learning_starts=50, log=False, seed=0)
    agent.train(
        total_timesteps=300,
        eval_env=env,
        ref_point=np.array(ref_point),
        timesteps_per_iter=100,
    )
    agent.save(save_replay_buffer=False, save_dir=str(tmp_path), filename="seed_0")

    ref_env = mo_gym.make(env_id)
    params = EvalParams(
        env_id=env_id,
        ref_point=ref_point,
        weights=equally_spaced_weights(2, n=5),
        ref_front=ref_env.unwrapped.pareto_front(gamma=0.99),
        gamma=0.99,
        num_episodes=1,
        on_missing=OnMissingBehavior.RAISE,
    )
    loader = class_loader(GPIPD, net_arch=net_arch, buffer_size=1_000, load_kwargs={"load_replay_buffer": False})

    results = evaluate_algorithm(tmp_path, params, loader)
    print_metrics(results)

    assert set(results) == {"seed_0"}
    assert results["seed_0"].front.shape[1] == 2
    assert results["seed_0"].metrics