import sqlite3


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


def test_unlinked_history_is_bounded_but_replay_linked_records_are_retained(tmp_path):
    memory = ExperienceMemory(tmp_path / "bounded.sqlite3", unlinked_limit=2)
    older = memory.record(agent="attacker", task="older", state_summary={}, action=0,
                          outcome="neutral", lesson="Older unlinked record.")
    newer = memory.record(agent="attacker", task="newer", state_summary={}, action=1,
                          outcome="neutral", lesson="Newer unlinked record.")
    # The periodic prune runs once every 1000 writes; emulate that boundary.
    memory._records_since_prune = 999
    current = memory.record(agent="attacker", task="current", state_summary={}, action=2,
                            outcome="neutral", lesson="Current record before replay attachment.")
    unlinked = memory.list(limit=10)
    assert {entry["id"] for entry in unlinked} == {newer["id"], current["id"]}
    assert all(entry["id"] != older["id"] for entry in unlinked)

    linked = memory.record(agent="defender", task="replay-linked", state_summary={}, action=0,
                           outcome="neutral", lesson="Retain this transition-backed record.")
    memory.remember_transition(agent="defender", context="1v1", state=[0.0], action=0,
                               reward=0.0, next_state=[0.1], terminal=False,
                               experience_id=linked["id"], capacity=2)
    # Reinitialize applies retention to persisted unlinked records, but must keep
    # the experience referenced by replay so it remains auditable and deletable.
    memory.close()
    reopened = ExperienceMemory(tmp_path / "bounded.sqlite3", unlinked_limit=2)
    ids = {entry["id"] for entry in reopened.list(limit=10)}
    assert linked["id"] in ids
    assert len([entry for entry in reopened.list(limit=10) if entry["id"] != linked["id"]]) <= 2
    reopened.close()


def test_unlinked_limit_configuration_is_validated(tmp_path):
    with pytest.raises(ValueError):
        ExperienceMemory(tmp_path / "bad-limit.sqlite3", unlinked_limit=0)
    with pytest.raises(ValueError):
        ExperienceMemory(tmp_path / "bad-limit-2.sqlite3", unlinked_limit=1_000_001)

def test_existing_legacy_schema_migrates_without_data_loss(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute(
        """CREATE TABLE experiences (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            agent TEXT NOT NULL,
            task TEXT NOT NULL,
            state_summary TEXT NOT NULL,
            action TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reward REAL,
            error_type TEXT,
            lesson TEXT NOT NULL,
            tags TEXT NOT NULL,
            metadata TEXT NOT NULL
        )"""
    )
    connection.execute(
        """INSERT INTO experiences
           (id, created_at, agent, task, state_summary, action, outcome, reward,
            error_type, lesson, tags, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "legacy-experience", "2026-01-01T00:00:00+00:00", "attacker",
            "legacy task", "{}", "{}", "failure", -1.0, None,
            "Legacy data must survive a schema migration.", "[]", "{}",
        ),
    )
    connection.commit()
    connection.close()

    memory = ExperienceMemory(db)
    entries = memory.list(limit=10)
    assert len(entries) == 1
    assert entries[0]["id"] == "legacy-experience"
    assert entries[0]["lesson_status"] == "active"
    assert memory.storage_status()["file_backed"] is True
    assert memory.storage_status()["path_explicitly_configured"] is True
    assert memory._connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert memory._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    memory.close()


def test_storage_status_does_not_expose_database_path(tmp_path):
    memory = ExperienceMemory(tmp_path / "private-path.sqlite3")
    status = memory.storage_status()
    assert status["backend"] == "sqlite"
    assert status["file_backed"] is True
    assert "path" not in status
    assert str(tmp_path) not in str(status)
    memory.close()
