"""Persistent, auditable episodic memory for CyberArena agents.

This is reflection/retrieval memory, not online policy learning: experiences are
recorded with outcomes and lessons; callers may retrieve relevant precedents.
SQLite is used so records survive restarts without an external service.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ExperienceMemory:
    """SQLite-backed store for task episodes, failures, and learned lessons."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        configured = path or os.environ.get("CYBERARENA_MEMORY_DB", "data/agent_memory.sqlite3")
        self.path = str(configured)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiences (
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
                    metadata TEXT NOT NULL,
                    lesson_status TEXT NOT NULL DEFAULT 'active'
                        CHECK (lesson_status IN ('active', 'deprecated'))
                );
                CREATE INDEX IF NOT EXISTS idx_experiences_created
                    ON experiences(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_experiences_agent
                    ON experiences(agent, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_experiences_outcome
                    ON experiences(outcome, created_at DESC);
                """
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record(
        self,
        *,
        agent: str,
        task: str,
        state_summary: Any,
        action: Any,
        outcome: str,
        lesson: str,
        reward: float | None = None,
        error_type: str | None = None,
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a completed or failed attempt and its explicit lesson."""
        required = {"agent": agent, "task": task, "outcome": outcome, "lesson": lesson}
        if any(not isinstance(v, str) or not v.strip() for v in required.values()):
            raise ValueError("agent, task, outcome, and lesson must be non-empty strings")
        if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
            raise ValueError("reward must be numeric or None")
        entry = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent.strip(),
            "task": task.strip(),
            "state_summary": _json(state_summary),
            "action": _json(action),
            "outcome": outcome.strip().lower(),
            "reward": float(reward) if reward is not None else None,
            "error_type": error_type,
            "lesson": lesson.strip(),
            "tags": _json(sorted({str(t).strip().lower() for t in tags if str(t).strip()})),
            "metadata": _json(metadata or {}),
        }
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO experiences
                (id, created_at, agent, task, state_summary, action, outcome,
                 reward, error_type, lesson, tags, metadata)
                VALUES (:id, :created_at, :agent, :task, :state_summary, :action,
                        :outcome, :reward, :error_type, :lesson, :tags, :metadata)""",
                entry,
            )
        return self._public(entry)

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        result = dict(row)
        for field in ("state_summary", "action", "tags", "metadata"):
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    pass
        return result

    def list(self, limit: int = 50, *, agent: str | None = None) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        query = "SELECT * FROM experiences"
        params: list[Any] = []
        if agent:
            query += " WHERE agent = ?"
            params.append(agent)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._public(row) for row in rows]

    def retrieve(self, query: str, *, agent: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve active precedents using deterministic lexical relevance.

        This intentionally avoids pretending to have semantic embeddings. A future
        embedding index can implement the same interface if evaluations justify it.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        tokens = {t for t in query.lower().split() if len(t) > 2}
        entries = self.list(limit=500, agent=agent)
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for entry in entries:
            if entry.get("lesson_status") != "active":
                continue
            corpus = " ".join([
                entry["task"], entry["outcome"], entry["lesson"],
                entry["error_type"] or "", " ".join(entry["tags"]),
                _json(entry["state_summary"]), _json(entry["action"]),
            ]).lower()
            score = sum(1 for token in tokens if token in corpus)
            if score:
                scored.append((score, entry["created_at"], entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [{**entry, "relevance_score": score} for score, _, entry in scored[:limit]]

    def update_lesson(
        self, experience_id: str, *, lesson: str | None = None,
        lesson_status: str | None = None,
    ) -> dict[str, Any]:
        """Correct/deprecate a lesson while preserving the auditable episode."""
        if lesson is not None and (not lesson.strip() or len(lesson) > 4000):
            raise ValueError("lesson must contain 1-4000 characters")
        if lesson_status not in (None, "active", "deprecated"):
            raise ValueError("lesson_status must be active or deprecated")
        updates, values = [], []
        if lesson is not None:
            updates.append("lesson = ?")
            values.append(lesson.strip())
        if lesson_status is not None:
            updates.append("lesson_status = ?")
            values.append(lesson_status)
        if not updates:
            raise ValueError("provide lesson and/or lesson_status")
        values.append(experience_id)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE experiences SET {', '.join(updates)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise KeyError(experience_id)
            row = self._connection.execute(
                "SELECT * FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        return self._public(row)

    def delete(self, experience_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM experiences WHERE id = ?", (experience_id,)
            )
            return cursor.rowcount > 0

    def reset(self) -> int:
        """Delete all stored experiences. UI/API must require explicit confirmation."""
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM experiences")
            return cursor.rowcount

    def summary(self) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """SELECT COUNT(*) AS total,
                   SUM(CASE WHEN outcome IN ('success', 'succeeded', 'completed') THEN 1 ELSE 0 END) AS successes,
                   SUM(CASE WHEN outcome IN ('failure', 'failed', 'error') THEN 1 ELSE 0 END) AS failures,
                   SUM(CASE WHEN lesson_status = 'active' THEN 1 ELSE 0 END) AS active_lessons
                   FROM experiences"""
            ).fetchone()
        total = row["total"] or 0
        successes = row["successes"] or 0
        failures = row["failures"] or 0
        return {
            "total_experiences": total,
            "successful_outcomes": successes,
            "failed_outcomes": failures,
            "active_lessons": row["active_lessons"] or 0,
            "observed_success_rate": successes / (successes + failures) if successes + failures else None,
            "note": "Observed stored outcomes only; this is not a controlled learning benchmark.",
        }
