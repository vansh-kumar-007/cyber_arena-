from __future__ import annotations

from utils.model_paths import resolve_model_dir


def test_model_dir_defaults_to_project_models_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("CYBERARENA_MODEL_DIR", raising=False)
    assert resolve_model_dir(tmp_path) == (tmp_path / "models").resolve()


def test_model_dir_uses_durable_mount_environment_override(tmp_path, monkeypatch):
    target = tmp_path / "persistent-volume" / "models"
    monkeypatch.setenv("CYBERARENA_MODEL_DIR", str(target))
    assert resolve_model_dir(tmp_path) == target.resolve()


def test_explicit_model_dir_takes_precedence_over_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CYBERARENA_MODEL_DIR", str(tmp_path / "env-models"))
    explicit = tmp_path / "test-models"
    assert resolve_model_dir(tmp_path, explicit) == explicit.resolve()


def test_blank_explicit_model_dir_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="cannot be blank"):
        resolve_model_dir(tmp_path, " ")
