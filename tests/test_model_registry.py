from __future__ import annotations

import json

import pytest

from utils.model_registry import ModelRegistry


def _candidate(registry: ModelRegistry, tmp_path, model_id: str) -> None:
    version_dir = tmp_path / "versions" / model_id
    version_dir.mkdir(parents=True)
    (version_dir / "attacker.pt").write_bytes(f"attacker-{model_id}".encode())
    (version_dir / "defender.pt").write_bytes(f"defender-{model_id}".encode())
    registry.register_candidate(
        model_id=model_id,
        attacker_path=f"versions/{model_id}/attacker.pt",
        defender_path=f"versions/{model_id}/defender.pt",
        training_run_id=f"run-{model_id}",
        training_config={"episodes": 100, "seed": 17},
        scenario_context="1v1",
    )


def _evaluate(registry: ModelRegistry, model_id: str) -> None:
    registry.record_evaluation(
        model_id,
        metrics={"attacker_win_rate": 0.7, "defender_win_rate": 0.2},
        compatible=True,
        passed=True,
        reason="matched-seed suite passed configured acceptance criteria",
    )


def test_registry_requires_evaluation_and_promotes_atomically(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _candidate(registry, tmp_path / "models", "stable-v1")
    with pytest.raises(ValueError, match="evaluated successfully"):
        registry.promote("stable-v1", reason="should not pass")

    _evaluate(registry, "stable-v1")
    active = registry.promote("stable-v1", reason="initial controlled test promotion")
    assert active["status"] == "active"
    pointer = json.loads(registry.active_path.read_text())
    assert pointer["model_id"] == "stable-v1"
    assert registry.active_model()["model_id"] == "stable-v1"


def test_registry_rollback_restores_previous_immutable_pair(tmp_path):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(model_dir)
    _candidate(registry, model_dir, "stable-v1")
    _evaluate(registry, "stable-v1")
    registry.promote("stable-v1", reason="known good baseline")

    _candidate(registry, model_dir, "candidate-v2")
    _evaluate(registry, "candidate-v2")
    registry.promote("candidate-v2", reason="controlled test candidate")
    assert registry.active_model()["model_id"] == "candidate-v2"

    rolled_back = registry.rollback(reason="synthetic regression in controlled test")
    assert rolled_back["model_id"] == "stable-v1"
    assert registry.active_model()["model_id"] == "stable-v1"
    assert registry.summary()["status_counts"]["rolled_back"] == 1


def test_registry_refuses_failed_evaluation_and_tampered_checkpoint(tmp_path):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(model_dir)
    _candidate(registry, model_dir, "rejected-v1")
    registry.record_evaluation(
        "rejected-v1", metrics={"attacker_win_rate": 0.0}, compatible=True,
        passed=False, reason="candidate failed regression threshold",
    )
    with pytest.raises(ValueError, match="evaluated successfully"):
        registry.promote("rejected-v1", reason="must be rejected")

    _candidate(registry, model_dir, "candidate-v2")
    _evaluate(registry, "candidate-v2")
    (model_dir / "versions" / "candidate-v2" / "attacker.pt").write_bytes(b"modified after evaluation")
    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.promote("candidate-v2", reason="tamper test")


def test_registry_rejects_checkpoint_path_escape(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="inside the configured model directory"):
        registry.register_candidate(
            model_id="escape",
            attacker_path="../outside.pt",
            defender_path="../outside.pt",
            training_run_id="run-escape",
            training_config={},
        )
