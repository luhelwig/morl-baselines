"""Tests for checkpoint discovery and loaders."""

from pathlib import Path

import mo_gymnasium as mo_gym
import pytest

from morl_reliable import class_loader, find_checkpoints, make_loader


class Recorder:
    """Minimal algorithm that records how the loader built and restored it."""

    def __init__(self, env, log=True, **kwargs):
        self.init_kwargs = {"log": log, **kwargs}

    def load(self, path, load_replay_buffer=True, strict=True):
        self.load_args = {"path": path, "load_replay_buffer": load_replay_buffer, "strict": strict}


class NoBufferFlag(Recorder):
    def load(self, path):
        self.load_args = {"path": path}


def _orbax_dir(path: Path, metadata="_CHECKPOINT_METADATA"):
    path.mkdir()
    (path / metadata).touch()


def test_names_keep_dots(tmp_path):
    (tmp_path / "lr0.001_seed0.tar").touch()
    (tmp_path / "lr0.001_seed1.tar").touch()
    assert list(find_checkpoints(tmp_path)) == ["lr0.001_seed0", "lr0.001_seed1"]

    jax_dir = tmp_path / "jax"
    jax_dir.mkdir()
    _orbax_dir(jax_dir / "gamma0.99_seed0")
    _orbax_dir(jax_dir / "gamma0.99_seed1", metadata="_METADATA")
    assert list(find_checkpoints(jax_dir)) == ["gamma0.99_seed0", "gamma0.99_seed1"]


def test_skips_non_checkpoints(tmp_path):
    _orbax_dir(tmp_path / "seed_0")
    _orbax_dir(tmp_path / "seed_1.orbax-checkpoint-tmp-1731")
    _orbax_dir(tmp_path / ".ipynb_checkpoints")
    (tmp_path / "wandb").mkdir()
    (tmp_path / "latest").symlink_to(tmp_path / "seed_0")
    assert find_checkpoints(tmp_path) == {"seed_0": tmp_path / "seed_0"}


def test_mixed_formats_need_a_pattern(tmp_path):
    (tmp_path / "seed_0.tar").touch()
    _orbax_dir(tmp_path / "seed_1")
    with pytest.raises(ValueError):
        find_checkpoints(tmp_path)
    assert list(find_checkpoints(tmp_path, "*.tar")) == ["seed_0"]


def test_pattern_names_are_relative_paths(tmp_path):
    for run in ("run_a", "run_b"):
        (tmp_path / run).mkdir()
        (tmp_path / run / "seed_0.pt").touch()
    assert list(find_checkpoints(tmp_path, "**/*.pt")) == ["run_a/seed_0", "run_b/seed_0"]


def test_no_checkpoints(tmp_path):
    (tmp_path / "wandb").mkdir()
    with pytest.raises(FileNotFoundError):
        find_checkpoints(tmp_path)


def test_class_loader_overrides_log_and_merges_load_kwargs():
    agent = class_loader(Recorder, load_kwargs={"strict": False}, log=True, net_arch=[8])(Path("x.tar"), None)
    assert agent.init_kwargs == {"log": False, "net_arch": [8]}
    assert agent.load_args == {"path": "x.tar", "load_replay_buffer": False, "strict": False}

    agent = class_loader(Recorder, load_kwargs={"load_replay_buffer": True})(Path("x.tar"), None)
    assert agent.load_args["load_replay_buffer"] is True

    agent = class_loader(NoBufferFlag)(Path("x.tar"), None)
    assert agent.load_args == {"path": "x.tar"}


def test_make_loader_accepts_name_class_and_loader():
    def custom(path, env):
        return None

    assert make_loader(custom) is custom
    assert callable(make_loader("envelope", {"net_arch": [8]}))
    assert make_loader(Recorder)(Path("x.tar"), None).init_kwargs == {"log": False}
    with pytest.raises(TypeError):
        make_loader(custom, algo_kwargs={"net_arch": [8]})
    with pytest.raises(TypeError):
        make_loader(42)


@pytest.mark.parametrize("name", ["gpi_ls_discrete_jax", "gpi_ls_continuous_jax"])
def test_jax_loader_without_seed_and_relative_path(name, tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from morl_reliable import get_algorithm

    env = mo_gym.make("deep-sea-treasure-v0" if "discrete" in name else "mo-mountaincarcontinuous-v0")
    get_algorithm(name)(env, log=False, net_arch=[8], buffer_size=10).save(save_dir=str(tmp_path), filename="seed_0")

    monkeypatch.chdir(tmp_path)
    agent = make_loader(name, {"net_arch": [8]})(Path("seed_0"), env)
    assert agent.seed is None
