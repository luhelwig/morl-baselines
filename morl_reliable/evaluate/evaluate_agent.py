from typing import Callable

from morl_baselines.common.morl_algorithm import MOAgent
import gymnasium as gym

def evaluate_agent(agent: MOAgent, env: gym.Env, **kwargs):
    """Evaluate an agent.
    Args:
        agent (MOAgent): An agent object.
    """