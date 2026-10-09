"""Small CPU-only integration smoke test for training, replay, and checkpoints."""
from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

import numpy as np

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
    # Short runs also write the separate latest-state pair required by resume=True.
    assert (tmp_path / "models" / "marl_2v2_attacker.pt").is_file()
    assert (tmp_path / "models" / "marl_2v2_defender.pt").is_file()
    reports=list((tmp_path/"models"/"reports").glob("training_2v2_seed-2026_*.json"))
    assert len(reports)==1
    report=json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["run_status"]=="completed"
    assert report["training_config"]["seed"]==2026
    assert report["training_config"]["episodes_completed_this_run"]==10
    assert report["metrics_summary"]["attacker_win_rate"] is not None
    assert len(report["checkpoint_artifacts"]["attacker_candidate"]["sha256"])==64
    assert len(report["checkpoint_artifacts"]["defender_candidate"]["sha256"])==64

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


def test_checkpoint_restores_safe_rng_state_and_replay_schedule(tmp_path):
    import torch

    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    original = DQNAttacker(state_size=37)
    original.epsilon = 0.23
    original.episode_count = 17
    original.memory.beta = 0.83
    checkpoint_path = tmp_path / "checkpoint.pt"
    original.save(str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_rng = checkpoint["rng_state"]
    python_saved = (
        saved_rng["python"]["version"],
        tuple(saved_rng["python"]["state"]),
        saved_rng["python"]["gauss"],
    )
    numpy_saved = saved_rng["numpy"]

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    restored = DQNAttacker(state_size=37)
    restored.load(str(checkpoint_path), restore_rng=True)

    assert random.getstate() == python_saved
    numpy_actual = np.random.get_state()
    assert numpy_actual[0] == numpy_saved["bit_generator"]
    np.testing.assert_array_equal(
        numpy_actual[1], np.asarray(numpy_saved["state"], dtype=np.uint32)
    )
    assert numpy_actual[2] == numpy_saved["position"]
    assert numpy_actual[3] == numpy_saved["has_gauss"]
    assert numpy_actual[4] == numpy_saved["cached_gaussian"]
    assert torch.equal(torch.get_rng_state(), saved_rng["torch_cpu"])
    assert restored.epsilon == pytest.approx(0.23)
    assert restored.episode_count == 17
    assert restored.memory.beta == pytest.approx(0.83)


def test_resume_fails_closed_when_checkpoint_pair_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(train_dqn, "PROJECT_ROOT", tmp_path)
    memory = ExperienceMemory(tmp_path / "memory.sqlite3")

    with pytest.raises(FileNotFoundError, match="both attacker and defender checkpoints"):
        train_dqn.train_marl(
            n_attackers=1,
            n_defenders=1,
            num_episodes=1,
            save_models=False,
            seed=19,
            memory=memory,
            resume=True,
        )

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    DQNAttacker(state_size=37).save(str(model_dir / "marl_1v1_attacker.pt"))
    with pytest.raises(FileNotFoundError, match="missing"):
        train_dqn.train_marl(
            n_attackers=1,
            n_defenders=1,
            num_episodes=1,
            save_models=False,
            seed=19,
            memory=memory,
            resume=True,
        )

    # A corrupt partner must raise rather than silently continuing from random weights.
    (model_dir / "marl_1v1_defender.pt").write_text("not a PyTorch checkpoint")
    with pytest.raises(RuntimeError, match="no fresh-weight fallback"):
        train_dqn.train_marl(
            n_attackers=1,
            n_defenders=1,
            num_episodes=1,
            save_models=False,
            seed=19,
            memory=memory,
            resume=True,
        )
    memory.close()


def test_stop_before_first_episode_never_writes_random_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(train_dqn, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_dqn, "_STOP_REQUESTED", True)
    memory = ExperienceMemory(tmp_path / "runtime" / "memory.sqlite3")
    output_dir = tmp_path / "separate-output"

    metrics, _, _ = train_dqn.train_marl(
        n_attackers=1,
        n_defenders=1,
        num_episodes=1,
        save_models=True,
        seed=123,
        memory=memory,
        restore_replay=False,
        output_dir=output_dir,
    )

    assert metrics.attacker_rewards == []
    assert metrics.defender_rewards == []
    assert not (output_dir / "marl_1v1_attacker.pt").exists()
    assert not (output_dir / "marl_1v1_defender.pt").exists()
    assert not (output_dir / "final_marl_attacker_1v1.pt").exists()
    assert not (output_dir / "final_marl_defender_1v1_defender.pt").exists()
    reports = list((output_dir / "reports").glob("training_1v1_seed-123_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["run_status"] == "interrupted"
    assert report["training_config"]["episodes_completed_this_run"] == 0
    assert report["checkpoint_artifacts"] == {}
    memory.close()
