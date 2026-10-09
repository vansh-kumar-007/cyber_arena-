import pytest

from utils.experience_memory import ExperienceMemory


def test_experience_persists_across_instances(tmp_path):
    db = tmp_path / "memory.sqlite3"
    first = ExperienceMemory(db)
    entry = first.record(
        agent="attacker",
        task="test a blocked node",
        state_summary={"node": "N1"},
        action={"name": "Phishing"},
        outcome="failure",
        reward=-1.5,
        error_type="blocked",
        lesson="A blocked node produced a negative immediate reward.",
        tags=["N1", "phishing"],
    )
    first.close()

    reopened = ExperienceMemory(db)
    entries = reopened.list()
    assert len(entries) == 1
    assert entries[0]["id"] == entry["id"]
    assert entries[0]["state_summary"] == {"node": "N1"}
    assert entries[0]["action"] == {"name": "Phishing"}
    assert reopened.summary()["failed_outcomes"] == 1
    reopened.close()


def test_retrieval_and_lesson_review_controls(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3")
    entry = memory.record(
        agent="defender",
        task="patch vulnerable database",
        state_summary={"node": "N2", "vulnerability": 0.9},
        action={"name": "Patch"},
        outcome="success",
        reward=2,
        lesson="Patching was followed by a positive reward in this episode.",
        tags=["database", "patch"],
    )
    matches = memory.retrieve("database patch vulnerability", agent="defender")
    assert matches
    assert matches[0]["id"] == entry["id"]
    assert matches[0]["relevance_score"] > 0

    updated = memory.update_lesson(entry["id"], lesson_status="deprecated")
    assert updated["lesson_status"] == "deprecated"
    assert memory.retrieve("database patch vulnerability", agent="defender") == []
    memory.update_lesson(entry["id"], lesson="Corrected, reviewed observation.", lesson_status="active")
    assert memory.retrieve("Corrected reviewed observation", agent="defender")
    memory.close()


def test_validation_delete_and_reset(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError):
        memory.record(agent="", task="x", state_summary={}, action={}, outcome="failure", lesson="x")
    with pytest.raises(ValueError):
        memory.list(limit=0)

    first = memory.record(agent="attacker", task="a", state_summary={}, action=0,
                          outcome="neutral", lesson="No measurable reward change.")
    second = memory.record(agent="defender", task="b", state_summary={}, action=1,
                           outcome="success", lesson="Positive reward observed.")
    assert memory.delete(first["id"]) is True
    assert memory.delete(first["id"]) is False
    assert memory.reset() == 1
    assert memory.list() == []
    memory.close()

def test_replay_transitions_persist_reload_and_respect_capacity(tmp_path):
    db = tmp_path / "replay.sqlite3"
    memory = ExperienceMemory(db)
    linked = memory.record(
        agent="attacker", task="replay example", state_summary={"step": 0},
        action={"id": 1}, outcome="failure", reward=-0.5,
        lesson="The action was followed by a negative reward.",
    )
    for idx in range(3):
        memory.remember_transition(
            agent="attacker",
            context="1v1",
            state=[float(idx), 0.0],
            action=idx,
            reward=-0.5 + idx,
            next_state=[float(idx + 1), 0.0],
            terminal=(idx == 2),
            experience_id=linked["id"] if idx == 0 else None,
            capacity=2,
        )
    transitions = memory.load_transitions(agent="attacker", context="1v1")
    assert [item["action"] for item in transitions] == [1, 2]
    assert memory.summary()["persisted_replay_transitions"] == 2
    memory.close()

    reopened = ExperienceMemory(db)
    assert reopened.load_transitions(agent="attacker", context="1v1") == transitions
    assert reopened.delete(linked["id"]) is True
    # The transition linked to the audit entry was already evicted by capacity;
    # removing an experience that still has linked replay should cascade.
    second = reopened.record(
        agent="defender", task="linked replay", state_summary={},
        action={"id": 0}, outcome="neutral", lesson="Neutral outcome observed.",
    )
    reopened.remember_transition(
        agent="defender", context="1v1", state=[0.0], action=0,
        reward=0.0, next_state=[0.1], terminal=False, experience_id=second["id"],
    )
    assert reopened.delete(second["id"]) is True
    assert reopened.load_transitions(agent="defender", context="1v1") == []
    assert reopened.reset() == 2
    assert reopened.summary()["persisted_replay_transitions"] == 0
    reopened.close()


def test_replay_rejects_invalid_transitions(tmp_path):
    memory = ExperienceMemory(tmp_path / "replay.sqlite3")
    with pytest.raises(ValueError):
        memory.remember_transition(
            agent="unknown", context="1v1", state=[0], action=0,
            reward=0, next_state=[0], terminal=False,
        )
    with pytest.raises(ValueError):
        memory.remember_transition(
            agent="attacker", context="1v1", state=[float("nan")], action=0,
            reward=0, next_state=[0], terminal=False,
        )
    with pytest.raises(ValueError):
        memory.remember_transition(
            agent="attacker", context="1v1", state=[0], action=12,
            reward=0, next_state=[0], terminal=False,
        )
    memory.close()
