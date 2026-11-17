"""Finding checkpoints on disk and rebuilding trained agents from them."""

import enum
import inspect
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import gymnasium as gym
import torch as th

from morl_reliable.algorithms import Algorithm, eval_init_kwargs, get_algorithm


Loader = Callable[[Path, gym.Env], Any]
"""Builds a trained agent from a checkpoint path and an environment."""

ORBAX_TMP_SUFFIX = ".orbax-checkpoint-tmp"
"""Part of the name of the directories Orbax writes before a checkpoint is finished."""

_ORBAX_METADATA_FILES = ("_CHECKPOINT_METADATA", "_METADATA")


def find_checkpoints(agents_dir: str | Path, pattern: str | None = None) -> dict[str, Path]:
    """Finds the checkpoints in a directory and names them.

    By default, the checkpoints are the ``*.tar`` files (PyTorch algorithms)
    or the finished Orbax checkpoint directories (JAX algorithms) directly in
    ``agents_dir``. With ``pattern``, they are the paths matching that glob.
    Either way, hidden paths, unfinished Orbax checkpoints and symlinks to a
    checkpoint already found are skipped.

    Args:
        agents_dir: Directory with the checkpoints of a single algorithm.
        pattern: Glob pattern relative to ``agents_dir``, e.g. ``"*.pt"``.

    Returns:
        Mapping from checkpoint name to path, sorted by name. The name is the
        path relative to ``agents_dir`` without the pattern's file extension
        (``.tar`` by default), e.g. ``"seed_0"``.

    Raises:
        FileNotFoundError: If no checkpoint is found.
        ValueError: If, without ``pattern``, ``agents_dir`` holds both
            ``.tar`` files and Orbax directories.
    """
    agents_dir = Path(agents_dir)
    if pattern is not None:
        found = _match(agents_dir, pattern)
    else:
        tar_files = _match(agents_dir, "*.tar")
        orbax_dirs = {name: path for name, path in _match(agents_dir, "*/").items() if _is_orbax_checkpoint(path)}
        if tar_files and orbax_dirs:
            raise ValueError(f"{agents_dir} has both .tar files and Orbax directories; choose with pattern=")
        found = tar_files or orbax_dirs
    if not found:
        raise FileNotFoundError(f"No {pattern or '*.tar or Orbax'} checkpoints found in {agents_dir}")
    return found


def _match(agents_dir: Path, pattern: str) -> dict[str, Path]:
    extension = PurePosixPath(pattern).suffix
    found: dict[str, Path] = {}
    seen: set[Path] = set()
    # Real paths first, so that a symlink is the one skipped when both lead to the same checkpoint.
    for path in sorted(agents_dir.glob(pattern), key=lambda p: (p.is_symlink(), p)):
        relative = path.relative_to(agents_dir)
        if any(part.startswith(".") for part in relative.parts) or ORBAX_TMP_SUFFIX in path.name:
            continue
        if (real := path.resolve()) in seen:
            continue
        seen.add(real)
        found[relative.as_posix().removesuffix(extension)] = path
    return dict(sorted(found.items()))


def _is_orbax_checkpoint(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _ORBAX_METADATA_FILES)


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
        load_kwargs: Extra arguments for ``agent.load``. ``load_replay_buffer=False``
            is added when ``load`` declares that parameter, unless given here.
        **init_kwargs: Constructor arguments; must match the training run. They
            override ``eval_init_kwargs(cls)``, and ``log`` is always False.

    Returns:
        A loader that rebuilds the agent with logging disabled.
    """
    init_kwargs = {**eval_init_kwargs(cls), **init_kwargs, "log": False}
    load_kwargs = dict(load_kwargs or {})
    if "load_replay_buffer" in inspect.signature(cls.load).parameters:
        load_kwargs.setdefault("load_replay_buffer", False)

    def load(path: Path, env: gym.Env) -> Any:
        agent = cls(env, **init_kwargs)
        agent.load(str(path), **load_kwargs)
        return agent

    return load


def make_loader(
    algorithm: "Algorithm | str | type | Loader", algo_kwargs: dict | None = None, load_kwargs: dict | None = None
) -> Loader:
    """Builds a loader from an algorithm name, an algorithm class or a loader.

    Args:
        algorithm: One of:
            - a name (``"envelope"``, ``Algorithm.ENVELOPE`` or a name
              registered with ``@register_algorithm``);
            - an algorithm class, used as in ``class_loader``;
            - a ``Loader``, returned as is.
        algo_kwargs: Constructor arguments; must match the training run.
        load_kwargs: Extra arguments for ``agent.load``, see ``class_loader``.

    Returns:
        A loader for the algorithm's checkpoints.

    Raises:
        ValueError: If ``algorithm`` is an unknown name.
        TypeError: If ``algorithm`` is none of the accepted kinds, or if
            ``algo_kwargs`` or ``load_kwargs`` are given with a ``Loader``.
    """
    if isinstance(algorithm, (str, enum.Enum)):
        algorithm = get_algorithm(algorithm)
    if inspect.isclass(algorithm):
        return class_loader(algorithm, load_kwargs, **(algo_kwargs or {}))
    if not callable(algorithm):
        raise TypeError(f"Expected an algorithm name, class or loader, got {type(algorithm).__name__}")
    if algo_kwargs or load_kwargs:
        raise TypeError("algo_kwargs and load_kwargs apply to an algorithm name or class, not to a Loader")
    return algorithm
