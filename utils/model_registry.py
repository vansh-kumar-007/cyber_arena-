"""Small, filesystem-backed model registry for single-instance inference deployments.

Registry metadata is not a substitute for persistent storage: place the registry and
immutable checkpoint files on the same durable volume as the serving process.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_VERSION = 1
ENVIRONMENT_VERSION = "cyberarena-network-env-v1"
FEATURE_SCHEMA_VERSION = "state-37-v1"
ACTION_SCHEMA_VERSION = "actions-12-v1"
ALLOWED_STATUSES = {"candidate", "evaluated", "approved", "active", "rejected", "rolled_back"}
_LOCK = threading.RLock()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ModelRegistry:
    """Registry with checksum verification, explicit evaluation gates and rollback."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self.registry_dir = self.model_dir / "registry"
        self.models_dir = self.model_dir / "versions"
        self.registry_path = self.registry_dir / "registry.json"
        self.active_path = self.registry_dir / "active.json"
        self.history_path = self.registry_dir / "history.jsonl"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            _atomic_json(self.registry_path, {"schema_version": REGISTRY_VERSION, "models": {}})

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("model registry is unreadable; refusing model lifecycle operation") from exc
        if value.get("schema_version") != REGISTRY_VERSION or not isinstance(value.get("models"), dict):
            raise RuntimeError("unsupported or invalid model registry schema")
        return value

    def _write(self, registry: dict[str, Any]) -> None:
        _atomic_json(self.registry_path, registry)

    def _resolve_checkpoint(self, relative: str) -> Path:
        path = (self.model_dir / relative).resolve()
        if not path.is_relative_to(self.model_dir):
            raise ValueError("checkpoint path must remain inside the configured model directory")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {relative}")
        return path

    def register_candidate(
        self,
        *,
        model_id: str,
        attacker_path: str,
        defender_path: str,
        training_run_id: str,
        training_config: dict[str, Any],
        scenario_context: str = "1v1",
        evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not model_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in model_id):
            raise ValueError("model_id may contain only letters, digits, dot, underscore and hyphen")
        if not training_run_id.strip():
            raise ValueError("training_run_id is required")
        attacker = self._resolve_checkpoint(attacker_path)
        defender = self._resolve_checkpoint(defender_path)
        metadata = {
            "model_id": model_id,
            "status": "candidate",
            "attacker_path": attacker.relative_to(self.model_dir).as_posix(),
            "defender_path": defender.relative_to(self.model_dir).as_posix(),
            "attacker_sha256": sha256_file(attacker),
            "defender_sha256": sha256_file(defender),
            "training_run_id": training_run_id,
            "training_config": training_config,
            "scenario_context": scenario_context,
            "environment_version": ENVIRONMENT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "created_at": _now(),
            "evaluation": evaluation,
            "decision_reason": "awaiting evaluation",
        }
        with _LOCK:
            registry = self._read()
            if model_id in registry["models"]:
                raise ValueError(f"model_id already exists: {model_id}")
            registry["models"][model_id] = metadata
            self._write(registry)
            self._event("candidate_registered", model_id, "candidate registered; not active")
        return metadata

    def _evaluation_acceptance_errors(
        self,
        registry: dict[str, Any],
        model: dict[str, Any],
        metrics: dict[str, Any],
        *,
        check_current_baseline: bool,
    ) -> list[str]:
        """Re-check versioned acceptance evidence at evaluation and promotion time.

        Candidate comparisons must compare both role-specific policies against the
        exact currently active stable pair, across a fixed multi-seed holdout suite.
        The initial registry bootstrap is a distinct, explicitly approved import of
        the unchanged legacy production pair; it is not a candidate improvement.
        """
        errors: list[str] = []
        if metrics.get("acceptance_criteria_version") != 1:
            errors.append("acceptance_criteria_version must be 1")
        if metrics.get("scenario_context") != model.get("scenario_context"):
            errors.append("evaluation scenario does not match the candidate scenario")
        kind = metrics.get("evaluation_kind")

        if kind == "baseline_bootstrap":
            if check_current_baseline and self.active_path.exists():
                errors.append("baseline bootstrap is only allowed before an active registry pointer exists")
            if metrics.get("source") != "existing_stable_checkpoint_pair":
                errors.append("baseline bootstrap must identify the existing stable checkpoint pair")
            if metrics.get("operator_approved") is not True:
                errors.append("baseline bootstrap requires explicit operator approval")
            if metrics.get("checkpoint_schema_verified") is not True:
                errors.append("baseline bootstrap requires checkpoint/schema compatibility verification")
            if metrics.get("attacker_checkpoint_sha256") != model.get("attacker_sha256"):
                errors.append("baseline bootstrap attacker checksum does not match registered bytes")
            if metrics.get("defender_checkpoint_sha256") != model.get("defender_sha256"):
                errors.append("baseline bootstrap defender checksum does not match registered bytes")
            return errors

        if kind != "candidate_comparison":
            errors.append("evaluation_kind must be baseline_bootstrap or candidate_comparison")
            return errors

        baseline_id = metrics.get("baseline_model_id")
        baseline = registry.get("models", {}).get(baseline_id) if isinstance(baseline_id, str) else None
        if not baseline:
            errors.append("candidate comparison must reference a registered stable baseline_model_id")
            return errors
        if baseline_id == model.get("model_id"):
            errors.append("candidate cannot be evaluated against itself")
        if baseline.get("status") not in {"active", "approved", "rolled_back"}:
            errors.append("baseline model must be active or previously approved")
        if baseline.get("scenario_context") != model.get("scenario_context"):
            errors.append("baseline and candidate scenario contexts differ")
        for key, expected in (
            ("environment_version", ENVIRONMENT_VERSION),
            ("feature_schema_version", FEATURE_SCHEMA_VERSION),
            ("action_schema_version", ACTION_SCHEMA_VERSION),
        ):
            if baseline.get(key) != expected:
                errors.append(f"baseline {key} is incompatible")

        for prefix, registered in (("baseline", baseline), ("candidate", model)):
            hashes = metrics.get(f"{prefix}_checkpoint_sha256")
            if not isinstance(hashes, dict):
                errors.append(f"{prefix}_checkpoint_sha256 evidence is required")
                continue
            expected_hashes = {
                "attacker": registered.get("attacker_sha256"),
                "defender": registered.get("defender_sha256"),
            }
            if hashes != expected_hashes:
                errors.append(f"{prefix} checkpoint checksums do not match the registered model")
            for path_key, hash_key in (("attacker_path", "attacker_sha256"), ("defender_path", "defender_sha256")):
                try:
                    checkpoint = self._resolve_checkpoint(registered[path_key])
                    if sha256_file(checkpoint) != registered[hash_key]:
                        errors.append(f"{prefix} checkpoint checksum verification failed: {path_key}")
                except (KeyError, OSError, ValueError) as exc:
                    errors.append(f"{prefix} checkpoint could not be verified: {path_key} ({type(exc).__name__})")

        if check_current_baseline:
            try:
                active = self.active_model()
            except Exception:
                active = None
                errors.append("active model could not be verified when the evaluation was recorded")
            if active is None or active.get("model_id") != baseline_id:
                errors.append("candidate must be evaluated against the current active stable model")

        for key in ("evaluation_suite_id", "evaluation_suite_version", "holdout_seed_set_sha256"):
            if not isinstance(metrics.get(key), str) or not metrics[key].strip():
                errors.append(f"{key} is required for reproducibility")

        for key, minimum in (("training_seed_count", 3), ("paired_episodes", 300)):
            value = metrics.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                errors.append(f"{key} must be an integer >= {minimum}")

        for key in (
            "attacker_crossplay_win_rate_delta_ci95",
            "defender_crossplay_success_rate_delta_ci95",
            "attacker_mean_reward_delta_ci95",
            "defender_mean_reward_delta_ci95",
        ):
            value = metrics.get(key)
            valid = (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(part, (int, float)) and not isinstance(part, bool) and math.isfinite(float(part))
                        for part in value)
                and float(value[0]) <= float(value[1])
            )
            if not valid:
                errors.append(f"{key} must contain finite [lower_95pct, upper_95pct] values")
            elif key in {"attacker_crossplay_win_rate_delta_ci95", "defender_crossplay_success_rate_delta_ci95"}:
                if float(value[0]) <= 0:
                    errors.append(f"{key} lower 95% confidence bound must be > 0")

        latency = metrics.get("inference_p95_latency_ms")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) < 0
            or float(latency) > 100.0
        ):
            errors.append("inference_p95_latency_ms must be finite and between 0 and 100 ms")

        safety = metrics.get("safety_checks")
        if not isinstance(safety, dict):
            errors.append("safety_checks evidence is required")
        else:
            if safety.get("schema_compatible") is not True:
                errors.append("candidate schema compatibility check did not pass")
            if safety.get("runtime_load_smoke_passed") is not True:
                errors.append("candidate runtime-load smoke test did not pass")
            for key in ("invalid_action_count", "non_finite_output_count", "critical_regression_count"):
                value = safety.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                    errors.append(f"safety_checks.{key} must be exactly zero")

        for key, expected in (
            ("environment_version", ENVIRONMENT_VERSION),
            ("feature_schema_version", FEATURE_SCHEMA_VERSION),
            ("action_schema_version", ACTION_SCHEMA_VERSION),
        ):
            if metrics.get(key) != expected:
                errors.append(f"evaluation {key} does not match the runtime schema")
        return errors

    def record_evaluation(
        self,
        model_id: str,
        *,
        metrics: dict[str, Any],
        compatible: bool,
        passed: bool,
        reason: str,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("evaluation decision reason is required")
        if not isinstance(metrics, dict):
            raise ValueError("evaluation metrics must be a structured object")
        with _LOCK:
            registry = self._read()
            model = registry["models"].get(model_id)
            if model is None:
                raise KeyError(model_id)
            if model["status"] in {"active", "rolled_back"}:
                raise ValueError("cannot overwrite evaluation metadata for an active/rolled-back model")

            if not compatible:
                acceptance_errors = ["checkpoint/environment/schema compatibility check failed"]
            elif not passed:
                acceptance_errors = ["evaluation did not pass its measured performance/safety checks"]
            else:
                acceptance_errors = self._evaluation_acceptance_errors(
                    registry, model, metrics, check_current_baseline=True
                )
            accepted = bool(compatible and passed and not acceptance_errors)
            model["evaluation"] = {
                "metrics": metrics,
                "compatible": bool(compatible),
                "passed": accepted,
                "claimed_passed": bool(passed),
                "reason": reason,
                "rejection_reasons": acceptance_errors,
                "evaluated_at": _now(),
            }
            model["status"] = "evaluated" if accepted else "rejected"
            model["decision_reason"] = reason if accepted else "rejected: " + "; ".join(acceptance_errors)
            self._write(registry)
            self._event("evaluation_recorded", model_id, model["decision_reason"])
            return model

    def _validated_model(self, registry: dict[str, Any], model_id: str) -> dict[str, Any]:
        model = registry["models"].get(model_id)
        if model is None:
            raise KeyError(model_id)
        if model["status"] not in {"evaluated", "approved", "active", "rolled_back"}:
            raise ValueError("model must be evaluated successfully before promotion")
        evaluation = model.get("evaluation") or {}
        if evaluation.get("compatible") is not True or evaluation.get("passed") is not True:
            raise ValueError("model does not have a passing compatible evaluation")
        acceptance_errors = self._evaluation_acceptance_errors(
            registry, model, evaluation.get("metrics") or {}, check_current_baseline=False
        )
        if acceptance_errors:
            raise ValueError("evaluation does not meet promotion acceptance criteria: " + "; ".join(acceptance_errors))
        if model.get("environment_version") != ENVIRONMENT_VERSION:
            raise ValueError("environment version mismatch")
        if model.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("feature schema mismatch")
        if model.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("action schema mismatch")
        for path_key, hash_key in (("attacker_path", "attacker_sha256"), ("defender_path", "defender_sha256")):
            path = self._resolve_checkpoint(model[path_key])
            if sha256_file(path) != model[hash_key]:
                raise ValueError(f"checkpoint checksum mismatch: {path_key}")
        return model

    def promote(self, model_id: str, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("promotion reason is required")
        with _LOCK:
            registry = self._read()
            model = self._validated_model(registry, model_id)
            previous = self.active_model()
            if previous and previous["model_id"] == model_id:
                return previous
            if previous:
                prior = registry["models"].get(previous["model_id"])
                if prior:
                    prior["status"] = "approved"
                    prior["decision_reason"] = f"superseded by {model_id}"
            model["status"] = "active"
            model["decision_reason"] = reason
            model["promoted_at"] = _now()
            # Immutable checkpoint files plus an atomic pointer replacement: inference
            # sees either the old complete pair or the new complete pair.
            _atomic_json(self.active_path, {
                "schema_version": REGISTRY_VERSION,
                "model_id": model_id,
                "attacker_path": model["attacker_path"],
                "defender_path": model["defender_path"],
                "attacker_sha256": model["attacker_sha256"],
                "defender_sha256": model["defender_sha256"],
                "environment_version": model["environment_version"],
                "feature_schema_version": model["feature_schema_version"],
                "action_schema_version": model["action_schema_version"],
                "promoted_at": model["promoted_at"],
            })
            self._write(registry)
            self._event("model_promoted", model_id, reason)
            return dict(model)

    def active_model(self) -> dict[str, Any] | None:
        if not self.active_path.exists():
            return None
        try:
            pointer = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("active model pointer is unreadable; refusing registry fallback") from exc
        if pointer.get("schema_version") != REGISTRY_VERSION:
            raise RuntimeError("unsupported active model pointer schema")
        registry = self._read()
        model = self._validated_model(registry, pointer.get("model_id", ""))
        if pointer.get("attacker_sha256") != model["attacker_sha256"] or pointer.get("defender_sha256") != model["defender_sha256"]:
            raise RuntimeError("active pointer and registry checksums disagree")
        return dict(model)

    def rollback(self, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("rollback reason is required")
        with _LOCK:
            current = self.active_model()
            if current is None:
                raise ValueError("cannot roll back: no active model exists")
            registry = self._read()
            previous_events = []
            if self.history_path.exists():
                for line in self.history_path.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "model_promoted" and event.get("model_id") != current["model_id"]:
                        previous_events.append(event.get("model_id"))
            previous_id = next((candidate for candidate in reversed(previous_events)
                                if candidate in registry["models"]
                                and registry["models"][candidate].get("status") in {"approved", "rolled_back"}), None)
            if previous_id is None:
                raise ValueError("no previously active model is available for rollback")
            current_record = registry["models"][current["model_id"]]
            current_record["status"] = "rolled_back"
            current_record["decision_reason"] = reason
            previous = self._validated_model(registry, previous_id)
            previous["status"] = "active"
            previous["decision_reason"] = reason
            previous["promoted_at"] = _now()
            _atomic_json(self.active_path, {
                "schema_version": REGISTRY_VERSION,
                "model_id": previous_id,
                "attacker_path": previous["attacker_path"],
                "defender_path": previous["defender_path"],
                "attacker_sha256": previous["attacker_sha256"],
                "defender_sha256": previous["defender_sha256"],
                "environment_version": previous["environment_version"],
                "feature_schema_version": previous["feature_schema_version"],
                "action_schema_version": previous["action_schema_version"],
                "promoted_at": previous["promoted_at"],
            })
            self._write(registry)
            self._event("model_rolled_back", previous_id, reason)
            return dict(previous)

    def _event(self, event: str, model_id: str, reason: str) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp": _now(), "event": event, "model_id": model_id, "reason": reason
            }, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def summary(self) -> dict[str, Any]:
        registry = self._read()
        active = self.active_model()
        counts: dict[str, int] = {}
        for model in registry["models"].values():
            status = model.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {
            "registry_schema_version": REGISTRY_VERSION,
            "model_count": len(registry["models"]),
            "status_counts": counts,
            "active_model_id": active["model_id"] if active else None,
            "registry_path": str(self.registry_path),
        }
