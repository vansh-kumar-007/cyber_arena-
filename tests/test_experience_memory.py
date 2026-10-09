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
