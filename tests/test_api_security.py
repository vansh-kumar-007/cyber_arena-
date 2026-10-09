import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient

from utils.experience_memory import ExperienceMemory


@pytest.fixture
def api_client(monkeypatch):
    fake_module = types.ModuleType("api.simulation")

    class FakeSimulationManager:
        def __init__(self):
            self.memory = ExperienceMemory(":memory:")
            self.memory.record(
                agent="attacker",
                task="api test episode",
                state_summary={"state": [0.0]},
                action={"id": 1},
                outcome="failure",
                reward=-1.0,
                lesson="This is a sample persisted observation.",
            )
            self.memory_write_errors = 0

        def status(self):
            return {
                "episode": 1,
                "step": 0,
                "is_done": False,
                "att_epsilon": 0.0,
                "def_epsilon": 0.0,
                "att_memory": 0,
                "def_memory": 0,
                "models_loaded": True,
                "active_model_id": None,
                "loaded_model_id": "legacy:test-attacker|test-defender",
                "loaded_policy_source": "legacy",
                "loaded_checkpoint_sha256": {"attacker": "a" * 64, "defender": "b" * 64},
                "policy_consistent": True,
                "model_registry_error": False,
                "state_size": 37,
                "n_attackers": 1,
                "n_defenders": 1,
                "experience_memory": self.memory.summary(),
                "memory_write_errors": self.memory_write_errors,
            }

        def reset(self, seed=None, n_attackers=None, n_defenders=None):
            return {
                "state": {}, "done": False, "step": 0, "episode": 2, "seed": seed,
                "n_attackers": n_attackers or 1, "n_defenders": n_defenders or 1,
                "models_loaded": True,
            }

        def step(self):
            return {"state": {}, "done": False, "step": 1, "episode": 1}

        def get_network_weights(self):
            return {"attacker": [], "defender": []}

    fake_module.SimulationManager = FakeSimulationManager
    monkeypatch.setitem(sys.modules, "api.simulation", fake_module)
    sys.modules.pop("api.main", None)
    api_module = importlib.import_module("api.main")
    with TestClient(api_module.app) as client:
        yield client, api_module
    api_module.sim.memory.close()
    sys.modules.pop("api.main", None)


def test_health_and_read_only_memory_routes(api_client, monkeypatch):
    client, _ = api_client
    monkeypatch.setenv("CYBERARENA_ADMIN_TOKEN", "test-secret")
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is True
    assert health.json()["readiness"] == "ready"
    assert health.json()["memory_write_errors"] == 0
    assert health.json()["state_size"] == 37
    assert health.json()["policy_consistent"] is True
    assert health.json()["loaded_policy_source"] == "legacy"
    assert health.json()["loaded_model_id"] == "legacy:test-attacker|test-defender"
    assert health.json()["model_registry_error"] is False

    assert client.get("/memory?limit=10").status_code == 403
    assert client.get("/memory/search?q=observation").status_code == 403
    headers = {"X-Admin-Token": "test-secret"}
    response = client.get("/memory?limit=10", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"]["summary"]["total_experiences"] == 1
    assert response.json()["data"]["experiences"][0]["lesson"] == "This is a sample persisted observation."
    assert client.get("/memory/search?q=observation", headers=headers).status_code == 200


def test_memory_mutations_require_server_token_and_reset_confirmation(api_client, monkeypatch):
    client, api_module = api_client
    entry_id = api_module.sim.memory.list()[0]["id"]
    monkeypatch.delenv("CYBERARENA_ADMIN_TOKEN", raising=False)
    payload = {"lesson": "Reviewed, corrected lesson."}
    assert client.patch(f"/memory/{entry_id}", json=payload).status_code == 503

    monkeypatch.setenv("CYBERARENA_ADMIN_TOKEN", "test-secret")
    assert client.patch(f"/memory/{entry_id}", json=payload, headers={"X-Admin-Token": "wrong"}).status_code == 403
    patched = client.patch(f"/memory/{entry_id}", json=payload, headers={"X-Admin-Token": "test-secret"})
    assert patched.status_code == 200
    assert patched.json()["data"]["lesson"] == "Reviewed, corrected lesson."

    assert client.delete(f"/memory/{entry_id}", headers={"X-Admin-Token": "test-secret"}).status_code == 200
    assert client.delete("/memory?confirm=false", headers={"X-Admin-Token": "test-secret"}).status_code == 400
    assert client.delete("/memory?confirm=true", headers={"X-Admin-Token": "test-secret"}).status_code == 200


def test_reset_forwards_seed_and_requested_team_size(api_client):
    client, _ = api_client
    response = client.post("/reset", json={"seed": 42, "n_attackers": 3, "n_defenders": 2})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["seed"] == 42
    assert data["n_attackers"] == 3
    assert data["n_defenders"] == 2
    assert data["models_loaded"] is True

    invalid = client.post("/reset", json={"n_attackers": 5, "n_defenders": 1})
    assert invalid.status_code == 422

def test_health_distinguishes_liveness_from_missing_trained_models(api_client, monkeypatch):
    client, api_module = api_client
    original_status = api_module.sim.status

    def status_without_models():
        return {**original_status(), "models_loaded": False}

    monkeypatch.setattr(api_module.sim, "status", status_without_models)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ready"] is False
    assert response.json()["readiness"] == "degraded"
    assert response.json()["models_loaded"] is False


def test_health_reports_memory_write_failures_as_not_ready(api_client):
    client, api_module = api_client
    original_status = api_module.sim.status

    def status_with_memory_errors():
        return {**original_status(), "memory_write_errors": 1}

    api_module.sim.status = status_with_memory_errors
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["readiness"] == "degraded"
    assert response.json()["memory_write_errors"] == 1



def test_health_marks_external_registry_promotion_not_ready_until_loaded(api_client, monkeypatch):
    client, api_module = api_client
    original_status = api_module.sim.status

    def status_with_pointer_mismatch():
        return {
            **original_status(),
            "active_model_id": "candidate-v2",
            "loaded_model_id": "stable-v1",
            "policy_consistent": False,
        }

    monkeypatch.setattr(api_module.sim, "status", status_with_pointer_mismatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["readiness"] == "degraded"
    assert response.json()["active_model_id"] == "candidate-v2"
    assert response.json()["loaded_model_id"] == "stable-v1"
    assert response.json()["policy_consistent"] is False


def test_step_and_weights_fail_closed_without_a_consistent_policy(api_client, monkeypatch):
    client, api_module = api_client
    original_status = api_module.sim.status
    def status_without_ready_policy():
        return {**original_status(), "models_loaded": False, "policy_consistent": False, "model_registry_error": False}
    monkeypatch.setattr(api_module.sim, "status", status_without_ready_policy)
    step = client.post("/step")
    weights = client.get("/weights")
    assert step.status_code == 503
    assert "will not serve randomly initialized weights" in step.json()["detail"]
    assert weights.status_code == 503
