"""Tests for the algorithm registry."""

import enum

import pytest

from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_reliable import (
    Algorithm,
    algorithms,
    get_algorithm,
    register_algorithm,
    unregister_algorithm,
)


@pytest.fixture
def register():
    """Registers a class under a name and unregisters it after the test."""
    names = set()

    def _register(name, cls):
        names.add(name)
        return register_algorithm(name)(cls)

    yield _register
    for name in names:
        unregister_algorithm(name)


def _make_class():
    class Algo:
        pass

    return Algo


@pytest.mark.parametrize("name", ["envelope", " ENVELOPE ", Algorithm.ENVELOPE])
def test_builtin_by_name(name):
    assert get_algorithm(name) is Envelope


def test_names_match_experiments_algos():
    pytest.importorskip("jax")
    from morl_baselines.common.experiments import ALGOS

    for member in Algorithm:
        assert get_algorithm(member) is ALGOS[member.value]


def test_other_enums_resolve_by_value():
    class CliAlgo(str, enum.Enum):
        ENVELOPE = "envelope"

    assert str(Algorithm.ENVELOPE) == "envelope"
    assert get_algorithm(CliAlgo.ENVELOPE) is Envelope


def test_register_and_unregister_user_algorithm():
    algo = register_algorithm("My_Algo")(_make_class())
    assert get_algorithm("my_algo") is algo
    unregister_algorithm("my_algo")
    with pytest.raises(ValueError):
        get_algorithm("my_algo")


def test_builtins_cannot_be_unregistered():
    with pytest.raises(ValueError):
        unregister_algorithm("envelope")


def test_registering_the_same_class_again_replaces_it(register):
    first, rerun = _make_class(), _make_class()
    register("rerun", first)
    register("rerun", rerun)
    assert get_algorithm("rerun") is rerun


def test_register_other_class_or_builtin_name_fails(register):
    register("dup", type("A", (), {}))
    with pytest.raises(ValueError):
        register_algorithm("dup")(type("B", (), {}))
    with pytest.raises(ValueError):
        register_algorithm("envelope")(type("C", (), {}))


def test_register_needs_a_name_and_a_class():
    with pytest.raises(TypeError):
        register_algorithm(_make_class())
    with pytest.raises(TypeError):
        register_algorithm("factory")(lambda env: None)


def test_unknown_name_lists_available():
    with pytest.raises(ValueError, match="envelope"):
        get_algorithm("envelop")


def test_eval_init_kwargs_only_for_builtins():
    assert algorithms.eval_init_kwargs(Envelope) == {"buffer_size": 1}
    assert algorithms.eval_init_kwargs(type("Sub", (Envelope,), {})) == {}
