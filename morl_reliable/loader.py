"""Loaders that rebuild trained agents from checkpoints."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch as th


Loader = Callable[[Path, gym.Env], Any]
"""Builds a trained agent from a checkpoint path and an environment."""


def pickle_loader(device: str | None = None) -> Loader:
    """Creates a loader for agents saved whole with ``th.save(agent)``.

    Args:
        device: Device to move the agent to. If None, keeps the device it
            was saved on.

    Returns:
        A loader that restores the agent and attaches the environment.
    """

    def load(path: Path, env: gym.Env) -> Any:
        agent = th.load(path, map_location=device, weights_only=False)
        if device is not None:
            agent.device = th.device(device)
        agent.env = env
        return agent

    return load


def class_loader(cls: type, load_kwargs: dict | None = None, **init_kwargs) -> Loader:
    """Creates a loader for agents built with ``cls(env, ...)`` and restored with ``load``.

    Args:
        cls: Algorithm class, e.g. ``Envelope``.
        load_kwargs: Extra arguments for ``agent.load``.
        **init_kwargs: Constructor arguments; must match the training run.

    Returns:
        A loader that rebuilds the agent with logging disabled.
    """
    load_kwargs = load_kwargs or {}

    def load(path: Path, env: gym.Env) -> Any:
        agent = cls(env, log=False, **init_kwargs)
        agent.load(str(path), **load_kwargs)
        return agent

    return load
