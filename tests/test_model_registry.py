from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.simulation as simulation_module
from api.simulation import SimulationManager
from utils.model_registry import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_VERSION,
    FEATURE_SCHEMA_VERSION,
    ModelRegistry,
)


def _candidate(registry: ModelRegistry, model_id: str) -> None:
    version_dir = registry.model_dir / "versions" / model_id
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


def _registered(registry: ModelRegistry, model_id: str) -> dict:
    return registry._read()["models"][model_id]


def _bootstrap_baseline(registry: ModelRegistry, model_id: str) -> None:
    model = _registered(registry, model_id)
    metrics = {
        "acceptance_criteria_version": 1,
        "evaluation_kind": "baseline_bootstrap",
        "scenario_context": model["scenario_context"],
        "source": "existing_stable_checkpoint_pair",
        "operator_approved": True,
        "checkpoint_schema_verified": True,
        "attacker_checkpoint_sha256": model["attacker_sha256"],
        "defender_checkpoint_sha256": model["defender_sha256"],
        "environment_version": ENVIRONMENT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    result = registry.record_evaluation(
        model_id,
        metrics=metrics,
        compatible=True,
        passed=True,
        reason="controlled stable-checkpoint bootstrap fixture",
    )
    assert result["evaluation"]["passed"] is True


def _comparison_metrics(
    registry: ModelRegistry,
    model_id: str,
    baseline_model_id: str = "stable-v1",
) -> dict:
    candidate = _registered(registry, model_id)
    baseline = _registered(registry, baseline_model_id)
    return {
        "acceptance_criteria_version": 1,
        "evaluation_kind": "candidate_comparison",
        "scenario_context": candidate["scenario_context"],
        "baseline_model_id": baseline_model_id,
        "evaluation_suite_id": "matched-1v1-cross-play",
        "evaluation_suite_version": "1",
        "holdout_seed_set_sha256": "a" * 64,
        "evaluation_seed_count": 3,
        "paired_episodes": 300,
        # Positive lower 95% confidence bounds are required for both cross-play
        # role metrics; rewards remain recorded with uncertainty for audit.
        "attacker_crossplay_win_rate_delta_ci95": [0.02, 0.08],
        "defender_crossplay_success_rate_delta_ci95": [0.01, 0.07],
        "attacker_mean_reward_delta_ci95": [-0.02, 0.12],
        "defender_mean_reward_delta_ci95": [-0.03, 0.09],
        "inference_p95_latency_ms": 24.0,
        "safety_checks": {
            "schema_compatible": True,
            "runtime_load_smoke_passed": True,
            "invalid_action_count": 0,
            "non_finite_output_count": 0,
            "critical_regression_count": 0,
        },
        "environment_version": ENVIRONMENT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "baseline_checkpoint_sha256": {
            "attacker": baseline["attacker_sha256"],
            "defender": baseline["defender_sha256"],
        },
        "candidate_checkpoint_sha256": {
            "attacker": candidate["attacker_sha256"],
            "defender": candidate["defender_sha256"],
        },
    }


def _evaluate_candidate(
    registry: ModelRegistry,
    model_id: str,
    baseline_model_id: str = "stable-v1",
    *,
    metrics_override: dict | None = None,
    passed: bool = True,
) -> dict:
    metrics = metrics_override or _comparison_metrics(registry, model_id, baseline_model_id)
    return registry.record_evaluation(
        model_id,
        metrics=metrics,
        compatible=True,
        passed=passed,
        reason="matched-seed cross-play suite acceptance decision",
    )


def _active_baseline(registry: ModelRegistry) -> None:
    _candidate(registry, "stable-v1")
    _bootstrap_baseline(registry, "stable-v1")
    registry.promote("stable-v1", reason="bootstrap unchanged stable policy in controlled fixture")


def test_registry_requires_evaluation_and_promotes_atomically(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _candidate(registry, "stable-v1")
    with pytest.raises(ValueError, match="evaluated successfully"):
        registry.promote("stable-v1", reason="must not promote before evaluation")

    _bootstrap_baseline(registry, "stable-v1")
    active = registry.promote("stable-v1", reason="initial controlled baseline bootstrap")
    assert active["status"] == "active"
    pointer = json.loads(registry.active_path.read_text())
    assert pointer["model_id"] == "stable-v1"
    assert registry.active_model()["model_id"] == "stable-v1"


def test_registry_rollback_restores_previous_immutable_pair(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)

    _candidate(registry, "candidate-v2")
    _evaluate_candidate(registry, "candidate-v2")
    registry.promote("candidate-v2", reason="measured cross-play improvement")
    assert registry.active_model()["model_id"] == "candidate-v2"

    rolled_back = registry.rollback(reason="synthetic regression in controlled test")
    assert rolled_back["model_id"] == "stable-v1"
    assert registry.active_model()["model_id"] == "stable-v1"
    assert registry.summary()["status_counts"]["rolled_back"] == 1


def test_registry_refuses_failed_evaluation_and_tampered_checkpoint(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)

    _candidate(registry, "rejected-v1")
    result = _evaluate_candidate(registry, "rejected-v1", passed=False)
    assert result["status"] == "rejected"
    assert result["evaluation"]["passed"] is False
    with pytest.raises(ValueError, match="evaluated successfully"):
        registry.promote("rejected-v1", reason="must be rejected")

    _candidate(registry, "candidate-v2")
    _evaluate_candidate(registry, "candidate-v2")
    (registry.model_dir / "versions" / "candidate-v2" / "attacker.pt").write_bytes(
        b"modified after evaluation"
    )
    with pytest.raises(ValueError, match="checksum"):
        registry.promote("candidate-v2", reason="tamper test")


def test_registry_rejects_missing_or_weak_acceptance_evidence(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)
    _candidate(registry, "candidate-v2")
    metrics = _comparison_metrics(registry, "candidate-v2")
    metrics["attacker_crossplay_win_rate_delta_ci95"] = [-0.01, 0.08]
    result = _evaluate_candidate(registry, "candidate-v2", metrics_override=metrics)
    assert result["status"] == "rejected"
    assert any("lower 95% confidence bound" in item for item in result["evaluation"]["rejection_reasons"])
    with pytest.raises(ValueError, match="evaluated successfully"):
        registry.promote("candidate-v2", reason="weak confidence bound")


def test_registry_rejects_stale_baseline_and_safety_failures(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)
    _candidate(registry, "candidate-v2")

    stale_baseline = _comparison_metrics(registry, "candidate-v2")
    stale_baseline["baseline_model_id"] = "missing-baseline"
    result = _evaluate_candidate(registry, "candidate-v2", metrics_override=stale_baseline)
    assert result["status"] == "rejected"
    assert any("registered stable baseline" in item for item in result["evaluation"]["rejection_reasons"])

    _candidate(registry, "candidate-v3")
    unsafe = _comparison_metrics(registry, "candidate-v3")
    unsafe["safety_checks"]["invalid_action_count"] = 1
    result = _evaluate_candidate(registry, "candidate-v3", metrics_override=unsafe)
    assert result["status"] == "rejected"
    assert any("invalid_action_count" in item for item in result["evaluation"]["rejection_reasons"])


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


def test_registry_fails_closed_on_corrupt_active_pointer_and_registry(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)
    good_pointer = registry.active_path.read_text(encoding="utf-8")

    registry.active_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="active model pointer is unreadable"):
        registry.active_model()

    registry.active_path.write_text(good_pointer, encoding="utf-8")
    registry.registry_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model registry is unreadable"):
        registry.active_model()



class _FakeAgent:
    def __init__(self, state_size: int):
        self.state_size = state_size
        self.epsilon = 0.0
        self.memory = []
        self.loaded_path: str | None = None

    def load(self, path: str) -> None:
        self.loaded_path = path


def _simulation_for_model_loading(model_dir: Path, monkeypatch) -> SimulationManager:
    monkeypatch.setattr(simulation_module, "DQNAttacker", _FakeAgent)
    monkeypatch.setattr(simulation_module, "DQNDefender", _FakeAgent)
    manager = object.__new__(SimulationManager)
    manager.env = SimpleNamespace(n_attackers=1, n_defenders=1, current_step=0)
    manager.state_size = 37
    manager.memory = SimpleNamespace(summary=lambda: {"total_experiences": 0})
    manager.model_dir = model_dir
    manager.loaded_model_id = None
    manager.loaded_policy_source = None
    manager.loaded_checkpoint_sha256 = {}
    manager.model_registry_error = False
    manager.models_loaded = False
    manager.episode_count = 0
    manager.is_done = False
    manager.memory_write_errors = 0
    manager.attacker = _FakeAgent(37)
    manager.defender = _FakeAgent(37)
    return manager


def _write_legacy_pair(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    attacker = model_dir / "final_marl_attacker_1v1.pt"
    defender = model_dir / "final_marl_defender_1v1_defender.pt"
    attacker.write_bytes(b"stable-attacker-fixture")
    defender.write_bytes(b"stable-defender-fixture")
    return attacker, defender


def test_promotion_rejects_candidate_evaluated_against_a_stale_baseline(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)
    _candidate(registry, "stale-v2")
    _evaluate_candidate(registry, "stale-v2")
    _candidate(registry, "newer-v2")
    _evaluate_candidate(registry, "newer-v2")

    registry.promote("newer-v2", reason="promote the freshly evaluated candidate")

    with pytest.raises(ValueError, match="current active stable model"):
        registry.promote("stale-v2", reason="stale evaluation must not be promoted")


def test_registry_rejects_impossible_crossplay_confidence_interval_bounds(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    _active_baseline(registry)
    _candidate(registry, "impossible-ci")
    metrics = _comparison_metrics(registry, "impossible-ci")
    metrics["attacker_crossplay_win_rate_delta_ci95"] = [2.0, 3.0]

    result = _evaluate_candidate(registry, "impossible-ci", metrics_override=metrics)

    assert result["status"] == "rejected"
    assert any("within [-1.0, 1.0]" in item for item in result["evaluation"]["rejection_reasons"])


def test_missing_active_pointer_after_promotion_refuses_legacy_fallback(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(model_dir)
    _candidate(registry, "stable-v1")
    _bootstrap_baseline(registry, "stable-v1")
    registry.promote("stable-v1", reason="controlled baseline activation")
    registry.active_path.unlink()
    _write_legacy_pair(model_dir)

    manager = _simulation_for_model_loading(model_dir, monkeypatch)
    manager.models_loaded = manager._load_models()

    assert manager.models_loaded is False
    assert manager.model_registry_error is True
    assert manager.loaded_model_id is None


def test_corrupt_registry_without_active_pointer_refuses_legacy_fallback(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    registry_dir = model_dir / "registry"
    registry_dir.mkdir(parents=True)
    (registry_dir / "registry.json").write_text("{broken", encoding="utf-8")
    _write_legacy_pair(model_dir)

    manager = _simulation_for_model_loading(model_dir, monkeypatch)
    manager.models_loaded = manager._load_models()

    assert manager.models_loaded is False
    assert manager.model_registry_error is True
    assert manager.loaded_model_id is None


def test_legacy_loader_records_checkpoint_identity_and_hashes(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    attacker, defender = _write_legacy_pair(model_dir)
    manager = _simulation_for_model_loading(model_dir, monkeypatch)

    manager.models_loaded = manager._load_models()

    assert manager.models_loaded is True
    assert manager.loaded_model_id == (
        "legacy:final_marl_attacker_1v1.pt|final_marl_defender_1v1_defender.pt"
    )
    assert manager.loaded_policy_source == "legacy"
    assert manager.loaded_checkpoint_sha256 == {
        "attacker": hashlib.sha256(attacker.read_bytes()).hexdigest(),
        "defender": hashlib.sha256(defender.read_bytes()).hexdigest(),
    }
    assert manager.status()["policy_consistent"] is True


def test_serving_status_detects_policy_pointer_changed_after_load(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    registry = ModelRegistry(model_dir)
    _active_baseline(registry)
    manager = _simulation_for_model_loading(model_dir, monkeypatch)
    manager.models_loaded = manager._load_models()

    assert manager.models_loaded is True
    before = manager.status()
    assert before["active_model_id"] == "stable-v1"
    assert before["loaded_model_id"] == "stable-v1"
    assert before["policy_consistent"] is True

    _candidate(registry, "candidate-v2")
    _evaluate_candidate(registry, "candidate-v2")
    registry.promote("candidate-v2", reason="simulate external pointer promotion")

    after = manager.status()
    assert after["active_model_id"] == "candidate-v2"
    assert after["loaded_model_id"] == "stable-v1"
    assert after["policy_consistent"] is False
