"""Small CPU-only integration smoke test for training, replay, and checkpoints."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import train_dqn
from agents.dqn_attacker import DQNAttacker
from utils.experience_memory import ExperienceMemory


def test_one_episode_trains_and_persists_replay_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train_dqn, "PROJECT_ROOT", tmp_path)
    memory = ExperienceMemory(tmp_path / "runtime" / "memory.sqlite3")
    metrics, attacker, defender = train_dqn.train_marl(
        n_attackers=2,
        n_defenders=2,
        num_episodes=1,
        save_models=True,
        seed=2026,
        memory=memory,
        restore_replay=True,
    )

    assert len(metrics.attacker_rewards) == 1
    assert len(metrics.defender_rewards) == 1
    assert len(attacker.memory) == 100
    assert len(defender.memory) == 100
    summary = memory.summary()
    assert summary["total_experiences"] == 200
    assert summary["persisted_replay_transitions"] == 200
    assert summary["replay_transitions_by_agent"] == {"attacker": 100, "defender": 100}

    attacker_path = tmp_path / "models" / "final_marl_attacker_2v2.pt"
    defender_path = tmp_path / "models" / "final_marl_defender_2v2_defender.pt"
    assert attacker_path.is_file()
    assert defender_path.is_file()

    restored = DQNAttacker(state_size=37)
    restored.load(str(attacker_path))
    assert restored.action_size == 12
    assert restored.state_size == 37
    replayed = train_dqn._restore_replay(
        memory, restored, agent_name="attacker", context="2v2"
    )
    assert replayed == 100
    assert len(restored.memory) == 100
    memory.close()
