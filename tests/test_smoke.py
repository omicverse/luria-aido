"""Smoke tests that run without the artifact bundle."""
import importlib
import pathlib

import pytest


def test_package_imports():
    m = importlib.import_module("luria_aido")
    assert m.__version__
    assert "Cell" in m.__all__


def test_config_defaults_are_relocatable(monkeypatch, tmp_path):
    from luria_aido import config

    monkeypatch.setenv("LURIA_AIDO_DATA", str(tmp_path))
    assert config.data_root() == tmp_path
    assert config.artifact("a", "b") == tmp_path / "a" / "b"


def test_missing_artifact_message_is_actionable(tmp_path):
    from luria_aido import config

    with pytest.raises(FileNotFoundError) as e:
        config.require(tmp_path / "nope.npy", "anchor table")
    msg = str(e.value)
    assert "LURIA_AIDO_DATA" in msg and "anchor table" in msg


def test_no_absolute_developer_paths_remain():
    """The package must not carry paths from the machine it was built on."""
    root = pathlib.Path(__file__).resolve().parents[1] / "luria_aido"
    offenders = [
        f"{p.relative_to(root)}:{i}"
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "/scratch/users/" in line or "/home/users/" in line
    ]
    assert not offenders, f"absolute developer paths: {offenders}"


def test_cuda_probe_never_raises():
    """cuda_ok() must answer False rather than propagate a CUDA error."""
    from luria_aido import config

    assert config.cuda_ok() in (True, False)
