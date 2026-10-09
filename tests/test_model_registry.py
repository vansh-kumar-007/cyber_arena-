from __future__ import annotations

import json

import pytest

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
        "training_seed_count": 3,
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

    stale_baseline = _comparison_metrics(registry, "candidate-v2", baseline_model_id="missing-baseline")
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
