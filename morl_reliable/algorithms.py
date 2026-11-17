"""Registry that maps algorithm names to their classes.

Built-in MORL-Baselines algorithms are listed in ``Algorithm`` and imported
only when requested, so evaluating one algorithm does not import the others
(or their dependencies, such as JAX). Algorithms implemented by the user are
added with ``@register_algorithm``.

Example:
    ```python
    @register_algorithm("my_algo")
    class MyAlgo(MOAgent, MOPolicy):
        ...

    get_algorithm("envelope")  # -> Envelope
    get_algorithm("my_algo")   # -> MyAlgo
    ```
"""

import enum
import inspect
import pkgutil
from typing import Any, NamedTuple


# Not enum.StrEnum (Python 3.11+): every morl_baselines algorithm imports this module.
class Algorithm(str, enum.Enum):
    """Built-in algorithms that can be evaluated by name.

    Only weight-conditioned algorithms are listed: the evaluation queries the
    policy with ``agent.eval(obs, w)`` for each preference weight vector. The
    names are the ones used by ``morl_baselines.common.experiments.ALGOS``.
    """

    ENVELOPE = "envelope"
    GPI_PD_DISCRETE = "gpi_pd_discrete"
    GPI_LS_DISCRETE = "gpi_ls_discrete"
    GPI_PD_CONTINUOUS = "gpi_pd_continuous"
    GPI_LS_CONTINUOUS = "gpi_ls_continuous"
    CAPQL = "capql"
    GPI_LS_DISCRETE_JAX = "gpi_ls_discrete_jax"
    GPI_LS_CONTINUOUS_JAX = "gpi_ls_continuous_jax"

    def __str__(self) -> str:
        """Returns the name, e.g. ``"envelope"``."""
        return self.value


class _Entry(NamedTuple):
    target: type | str  # a registered class, or the "module:Class" path of a built-in
    eval_init_kwargs: dict[str, Any]


# Constructor arguments that skip what only training needs when a built-in is
# rebuilt for evaluation. The GPI-LS classes set dyna=False themselves and
# reject the argument.
_NO_BUFFER = {"buffer_size": 1}
_NO_BUFFER_NO_DYNA = {"buffer_size": 1, "dyna": False}

_REGISTRY: dict[str, _Entry] = {
    Algorithm.ENVELOPE.value: _Entry("morl_baselines.multi_policy.envelope.envelope:Envelope", _NO_BUFFER),
    Algorithm.GPI_PD_DISCRETE.value: _Entry("morl_baselines.multi_policy.gpi_pd.gpi_pd:GPIPD", _NO_BUFFER_NO_DYNA),
    Algorithm.GPI_LS_DISCRETE.value: _Entry("morl_baselines.multi_policy.gpi_pd.gpi_pd:GPILS", _NO_BUFFER),
    Algorithm.GPI_PD_CONTINUOUS.value: _Entry(
        "morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action:GPIPDContinuousAction", _NO_BUFFER_NO_DYNA
    ),
    Algorithm.GPI_LS_CONTINUOUS.value: _Entry(
        "morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action:GPILSContinuousAction", _NO_BUFFER
    ),
    Algorithm.CAPQL.value: _Entry("morl_baselines.multi_policy.capql.capql:CAPQL", _NO_BUFFER),
    Algorithm.GPI_LS_DISCRETE_JAX.value: _Entry("morl_baselines.multi_policy.gpi_ls_jax.gpi_ls_jax:GPILS", _NO_BUFFER),
    Algorithm.GPI_LS_CONTINUOUS_JAX.value: _Entry(
        "morl_baselines.multi_policy.gpi_ls_jax.gpi_ls_continuous_action_jax:GPILSContinuousAction", _NO_BUFFER
    ),
}


def _normalize(name: "Algorithm | str") -> str:
    if isinstance(name, enum.Enum):
        name = name.value
    if not isinstance(name, str):
        raise TypeError(f"Algorithm names must be strings, got {type(name).__name__}")
    return name.strip().lower()


def _path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def register_algorithm(name: str):
    """Decorator that registers an algorithm class under ``name``.

    The class must:

    - be built with ``cls(env, **kwargs)`` and accept ``log=False``;
    - restore a checkpoint with ``agent.load(path, ...)``;
    - return the greedy action with ``agent.eval(obs, w)`` for a weight vector ``w``;
    - have a ``gamma`` attribute, unless ``EvalParams.gamma`` is set.

    Registering a class with the same module and name again (e.g. when a
    notebook cell is re-run) replaces the previous one.

    Args:
        name: Unique name of the algorithm (case-insensitive).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        TypeError: If ``name`` is not a string (e.g. ``@register_algorithm``
            without parentheses) or the decorated object is not a class.
        ValueError: If ``name`` is a built-in or is registered to another class.

    Example:
        ```python
        @register_algorithm("my_algo")
        class MyAlgo(MOAgent, MOPolicy):
            ...
        ```
    """
    if not isinstance(name, str):
        raise TypeError('register_algorithm needs a name, e.g. @register_algorithm("my_algo")')
    key = _normalize(name)

    def decorator(cls: type) -> type:
        if not inspect.isclass(cls):
            raise TypeError(f"Only classes can be registered, got {cls!r}")
        entry = _REGISTRY.get(key)
        if entry is not None and (isinstance(entry.target, str) or _path(entry.target) != _path(cls)):
            raise ValueError(f"Algorithm {name} already registered")
        _REGISTRY[key] = _Entry(cls, {})
        return cls

    return decorator


def unregister_algorithm(name: str) -> None:
    """Removes an algorithm added with ``register_algorithm``.

    Args:
        name: Name it was registered under (case-insensitive).

    Raises:
        ValueError: If ``name`` is a built-in or is not registered.
    """
    key = _normalize(name)
    entry = _REGISTRY.get(key)
    if entry is None or isinstance(entry.target, str):
        raise ValueError(f"No user-registered algorithm named {name!r}")
    del _REGISTRY[key]


def get_algorithm(name: "Algorithm | str") -> type:
    """Resolves an algorithm class by name.

    Args:
        name: An ``Algorithm`` member or a registered name (case-insensitive).

    Returns:
        The algorithm class.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If no algorithm with the given name exists.
    """
    try:
        target = _REGISTRY[_normalize(name)].target
    except KeyError:
        raise ValueError(f"Unknown algorithm {str(name)!r}. Available: {', '.join(sorted(_REGISTRY))}") from None
    return pkgutil.resolve_name(target) if isinstance(target, str) else target


def eval_init_kwargs(cls: type) -> dict[str, Any]:
    """Returns the constructor arguments that rebuild ``cls`` for evaluation only.

    For a built-in algorithm they skip what only training needs, such as the
    replay buffer. For any other class they are empty.

    Args:
        cls: Algorithm class.

    Returns:
        A new dict of constructor arguments.
    """
    path = _path(cls)
    for target, kwargs in _REGISTRY.values():
        if target is cls or target == path:
            return dict(kwargs)
    return {}
