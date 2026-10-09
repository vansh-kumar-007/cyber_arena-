"""Small CPU-only integration smoke test for training, replay, and checkpoints."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import train_dqn
from agents.dqn_attacker import DQNAttacker
from api.simulation import SimulationManager
from utils.experience_memory import ExperienceMemory


def test_short_run_trains_and_persists_replay_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train_dqn, "PROJECT_ROOT", tmp_path)
    memory = ExperienceMemory(tmp_path / "runtime" / "memory.sqlite3")
    metrics, attacker, defender = train_dqn.train_marl(
        n_attackers=2,
        n_defenders=2,
        num_episodes=10,
        save_models=True,
        seed=2026,
        memory=memory,
        restore_replay=True,
    )

    assert len(metrics.attacker_rewards) == 10
    assert len(metrics.defender_rewards) == 10
    assert attacker.losses, "attacker DQN did not perform a gradient update"
    assert defender.losses, "defender DQN did not perform a gradient update"
    assert len(attacker.memory) >= attacker.batch_size
    assert len(defender.memory) >= defender.batch_size
    summary = memory.summary()
    assert summary["total_experiences"] > 0
    assert summary["persisted_replay_transitions"] == summary["total_experiences"]
    assert summary["replay_transitions_by_agent"]["attacker"] == len(attacker.memory)
    assert summary["replay_transitions_by_agent"]["defender"] == len(defender.memory)

    attacker_path = tmp_path / "models" / "final_marl_attacker_2v2.pt"
    defender_path = tmp_path / "models" / "final_marl_defender_2v2_defender.pt"
    assert attacker_path.is_file()
    assert defender_path.is_file()

    # The API must load the requested scenario's checkpoint pair.
    manager = SimulationManager(memory=memory, model_dir=tmp_path / "models", seed=2026)
    selected = manager.reset(seed=2026, n_attackers=2, n_defenders=2)
    assert selected["n_attackers"] == 2
    assert selected["n_defenders"] == 2
    assert selected["state_size"] == 37
    assert selected["models_loaded"] is True

    restored = DQNAttacker(state_size=37)
    restored.load(str(attacker_path))
    assert restored.action_size == 12
    assert restored.state_size == 37
    replayed = train_dqn._restore_replay(
        memory, restored, agent_name="attacker", context="2v2"
    )
    assert replayed == summary["replay_transitions_by_agent"]["attacker"]
    assert len(restored.memory) == replayed
    memory.close()
