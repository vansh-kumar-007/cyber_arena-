"""Auditable local episodic memory for CyberArena agents.

This is reflection/retrieval memory, not online policy learning. SQLite provides
transactional local storage; restart/deploy durability depends on the host filesystem
and is deliberately not assumed.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ExperienceMemory:
    """SQLite-backed store for task episodes, failures, and learned lessons."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        unlinked_limit: int | None = None,
    ) -> None:
        environment_path = os.environ.get("CYBERARENA_MEMORY_DB", "").strip()
        configured = path if path is not None else environment_path
        self.path_explicitly_configured = path is not None or bool(environment_path)
        if configured is None or str(configured).strip() == "":
            configured = Path(__file__).resolve().parents[1] / "data" / "agent_memory.sqlite3"
        self.path = ":memory:" if str(configured) == ":memory:" else str(Path(configured).expanduser())
        self.file_backed = self.path != ":memory:"
        configured_limit = unlinked_limit
        if configured_limit is None:
            configured_limit = int(os.environ.get("CYBERARENA_UNLINKED_EXPERIENCE_LIMIT", "20000"))
        if (not isinstance(configured_limit, int) or isinstance(configured_limit, bool)
                or not 1 <= configured_limit <= 1_000_000):
            raise ValueError("unlinked_limit must be an integer between 1 and 1000000")
        self.unlinked_record_limit = configured_limit
        self._records_since_prune = 0
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            current_schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if current_schema_version > SCHEMA_VERSION:
                self._connection.close()
                raise RuntimeError(
                    f"memory database schema version {current_schema_version} is newer than "
                    f"supported version {SCHEMA_VERSION}"
                )
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
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
                CREATE TABLE IF NOT EXISTS replay_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    agent TEXT NOT NULL CHECK (agent IN ('attacker', 'defender')),
                    context TEXT NOT NULL,
                    experience_id TEXT REFERENCES experiences(id) ON DELETE CASCADE,
                    state TEXT NOT NULL,
                    action INTEGER NOT NULL CHECK (action >= 0),
                    reward REAL NOT NULL,
                    next_state TEXT NOT NULL,
                    terminal INTEGER NOT NULL CHECK (terminal IN (0, 1))
                );
                CREATE INDEX IF NOT EXISTS idx_replay_context
                    ON replay_transitions(agent, context, id DESC);
                CREATE INDEX IF NOT EXISTS idx_replay_experience
                    ON replay_transitions(experience_id);
                """
            )
            # Apply additive migrations for databases created before the current schema.
            # Existing experiences are retained and receive an active lesson status.
            experience_columns = {
                row["name"] for row in self._connection.execute("PRAGMA table_info(experiences)")
            }
            if "lesson_status" not in experience_columns:
                self._connection.execute(
                    "ALTER TABLE experiences ADD COLUMN lesson_status TEXT NOT NULL "
                    "DEFAULT 'active' CHECK (lesson_status IN ('active', 'deprecated'))"
                )
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._connection.commit()
            integrity = self._connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                self._connection.close()
                raise RuntimeError(f"memory database integrity check failed: {integrity}")
            # Trim an existing database at startup. Replay-linked episodes are retained.
            with self._connection:
                self._prune_unlinked_locked()

    def _prune_unlinked_locked(self, *, exclude_id: str | None = None) -> int:
        """Keep only the newest unlinked episodes; caller holds the connection lock."""
        if exclude_id is None:
            cursor = self._connection.execute(
                """DELETE FROM experiences
                   WHERE id IN (
                       SELECT e.id FROM experiences AS e
                       WHERE NOT EXISTS (
                           SELECT 1 FROM replay_transitions AS r WHERE r.experience_id = e.id
                       )
                       ORDER BY e.created_at DESC, e.id DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (self.unlinked_record_limit,),
            )
        else:
            # Preserve the just-inserted row until the caller has a chance to attach
            # its replay transition, even when this write triggers pruning.
            cursor = self._connection.execute(
                """DELETE FROM experiences
                   WHERE id IN (
                       SELECT e.id FROM experiences AS e
                       WHERE e.id != ? AND NOT EXISTS (
                           SELECT 1 FROM replay_transitions AS r WHERE r.experience_id = e.id
                       )
                       ORDER BY e.created_at DESC, e.id DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (exclude_id, max(0, self.unlinked_record_limit - 1)),
            )
        return cursor.rowcount

    def storage_status(self) -> dict[str, Any]:
        """Expose storage facts without implying that the host filesystem is durable."""
        return {
            "backend": "sqlite",
            "file_backed": self.file_backed,
            "path_explicitly_configured": self.path_explicitly_configured,
            "schema_version": SCHEMA_VERSION,
            "persistence_verified": False,
            "persistence_status": "unverified",
            "durability_note": (
                "File-backed SQLite is only as durable as the host filesystem. "
                "Survival across redeploys/restarts has not been verified."
                if self.file_backed
                else "In-memory SQLite records are lost when the process restarts."
            ),
        }

    def export_json(self, destination: str | os.PathLike[str], *, overwrite: bool = False) -> dict[str, Any]:
        """Atomically export a versioned JSON package without overwriting by default."""
        target = Path(destination).expanduser().resolve()
        if self.file_backed and target == Path(self.path).expanduser().resolve():
            raise ValueError("export destination must not be the active SQLite database")
        if target.exists() and not overwrite:
            raise FileExistsError(f"export already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                experiences = [
                    self._public(row) for row in self._connection.execute(
                        "SELECT * FROM experiences ORDER BY created_at, id"
                    ).fetchall()
                ]
                transitions = []
                for row in self._connection.execute("SELECT * FROM replay_transitions ORDER BY id").fetchall():
                    item = dict(row)
                    item["state"] = json.loads(item["state"])
                    item["next_state"] = json.loads(item["next_state"])
                    item["terminal"] = bool(item["terminal"])
                    transitions.append(item)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        package = {
            "format": "cyberarena.experience-memory", "format_version": 1,
            "schema_version": SCHEMA_VERSION, "exported_at": datetime.now(timezone.utc).isoformat(),
            "experiences": experiences, "replay_transitions": transitions,
        }
        try:
            content = json.dumps(package, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError("memory contains non-JSON or non-finite values; export refused") from exc
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            if overwrite:
                os.replace(temporary_path, target)
                temporary_path = None
            else:
                os.link(temporary_path, target)
                temporary_path.unlink()
                temporary_path = None
        finally:
            if temporary_path is not None:
                try: temporary_path.unlink()
                except FileNotFoundError: pass
        return {
            "path": str(target), "experiences_exported": len(experiences),
            "replay_transitions_exported": len(transitions), "format_version": 1,
        }

    def import_json(self, source: str | os.PathLike[str], *, max_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
        """Validate and transactionally merge a compatible memory package."""
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        source_path = Path(source).expanduser().resolve()
        if self.file_backed and source_path == Path(self.path).expanduser().resolve():
            raise ValueError("import source must not be the active SQLite database")
        try:
            with source_path.open("rb") as stream: raw = stream.read(max_bytes + 1)
        except OSError as exc: raise ValueError(f"cannot read import package: {exc}") from exc
        if len(raw) > max_bytes: raise ValueError(f"import package exceeds the {max_bytes}-byte limit")
        def reject_constant(value: str): raise ValueError(f"non-finite JSON constant is not allowed: {value}")
        try:
            package = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ValueError("import package is not valid strict UTF-8 JSON") from exc
        if not isinstance(package, dict): raise ValueError("import package root must be an object")
        if package.get("format") != "cyberarena.experience-memory" or package.get("format_version") != 1:
            raise ValueError("unsupported experience-memory package format or version")
        if package.get("schema_version") != SCHEMA_VERSION: raise ValueError("experience-memory schema is incompatible")
        experiences, transitions = package.get("experiences"), package.get("replay_transitions")
        if not isinstance(experiences, list) or not isinstance(transitions, list):
            raise ValueError("experiences and replay_transitions must be arrays")
        if len(experiences) > 100_000 or len(transitions) > 100_000:
            raise ValueError("package exceeds the supported record-count limit")
        exp_fields = {
            "id", "created_at", "agent", "task", "state_summary", "action", "outcome",
            "reward", "error_type", "lesson", "tags", "metadata", "lesson_status",
        }
        incoming_ids: set[str] = set()
        normalized_exp = []
        for i, row in enumerate(experiences):
            if not isinstance(row, dict) or set(row) != exp_fields:
                raise ValueError(f"experience[{i}] has an incompatible shape")
            for key, limit in (("id",128),("created_at",64),("agent",100),("task",1000),("outcome",50),("lesson",4000)):
                val = row[key]
                if not isinstance(val, str) or not val.strip() or len(val) > limit:
                    raise ValueError(f"experience[{i}].{key} is invalid")
            if row["id"] in incoming_ids: raise ValueError(f"duplicate experience ID in package: {row['id']}")
            incoming_ids.add(row["id"])
            if row["error_type"] is not None and (not isinstance(row["error_type"],str) or len(row["error_type"])>250):
                raise ValueError(f"experience[{i}].error_type is invalid")
            reward = row["reward"]
            if reward is not None and (isinstance(reward,bool) or not isinstance(reward,(int,float)) or not math.isfinite(float(reward))):
                raise ValueError(f"experience[{i}].reward must be finite or null")
            if not isinstance(row["tags"],list) or len(row["tags"])>200 or any(not isinstance(t,str) or len(t)>200 for t in row["tags"]):
                raise ValueError(f"experience[{i}].tags is invalid")
            if not isinstance(row["metadata"],dict) or row["lesson_status"] not in ("active","deprecated"):
                raise ValueError(f"experience[{i}] metadata/lesson_status is invalid")
            for key in ("state_summary","action","tags","metadata"):
                try: json.dumps(row[key],allow_nan=False)
                except (TypeError,ValueError,RecursionError) as exc: raise ValueError(f"experience[{i}].{key} is not strict JSON") from exc
            normalized_exp.append((
                row["id"],row["created_at"],row["agent"],row["task"],_json(row["state_summary"]),_json(row["action"]),
                row["outcome"],float(reward) if reward is not None else None,row["error_type"],row["lesson"],
                _json(row["tags"]),_json(row["metadata"]),row["lesson_status"],
            ))
        transition_fields = {"id","created_at","agent","context","experience_id","state","action","reward","next_state","terminal"}
        normalized_transitions = []
        def finite(value):
            return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(float(value))
        for i, row in enumerate(transitions):
            if not isinstance(row,dict) or set(row)!=transition_fields: raise ValueError(f"replay_transition[{i}] has an incompatible shape")
            if isinstance(row["id"],bool) or not isinstance(row["id"],int) or row["id"]<1: raise ValueError(f"replay_transition[{i}].id is invalid")
            if not isinstance(row["created_at"],str) or not row["created_at"] or len(row["created_at"])>64: raise ValueError(f"replay_transition[{i}].created_at is invalid")
            if row["agent"] not in ("attacker","defender"): raise ValueError(f"replay_transition[{i}].agent is invalid")
            context=row["context"]
            if not isinstance(context,str) or not context.strip() or len(context)>128: raise ValueError(f"replay_transition[{i}].context is invalid")
            linked=row["experience_id"]
            if linked is not None and (not isinstance(linked,str) or not linked.strip() or len(linked)>128): raise ValueError(f"replay_transition[{i}].experience_id is invalid")
            state,nxt=row["state"],row["next_state"]
            if not isinstance(state,list) or not isinstance(nxt,list) or not state or len(state)!=len(nxt) or len(state)>4096:
                raise ValueError(f"replay_transition[{i}] state dimensions are invalid")
            if not all(finite(v) for v in state+nxt): raise ValueError(f"replay_transition[{i}] has non-finite states")
            action,reward,terminal=row["action"],row["reward"],row["terminal"]
            if isinstance(action,bool) or not isinstance(action,int) or not 0<=action<12: raise ValueError(f"replay_transition[{i}].action is invalid")
            if not finite(reward) or not isinstance(terminal,bool): raise ValueError(f"replay_transition[{i}] reward/terminal is invalid")
            normalized_transitions.append((row["created_at"],row["agent"],context,linked,_json([float(v) for v in state]),action,float(reward),_json([float(v) for v in nxt]),int(terminal)))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing={item[0] for item in self._connection.execute("SELECT id FROM experiences").fetchall()}
                collisions=incoming_ids & existing
                if collisions: raise ValueError(f"experience ID already exists; import refused: {sorted(collisions)[0]}")
                allowed=incoming_ids | existing
                for i,row in enumerate(transitions):
                    if row["experience_id"] is not None and row["experience_id"] not in allowed:
                        raise ValueError(f"replay_transition[{i}] references a missing experience ID")
                self._connection.executemany(
                    """INSERT INTO experiences (id,created_at,agent,task,state_summary,action,outcome,reward,error_type,lesson,tags,metadata,lesson_status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", normalized_exp)
                self._connection.executemany(
                    """INSERT INTO replay_transitions (created_at,agent,context,experience_id,state,action,reward,next_state,terminal)
                       VALUES (?,?,?,?,?,?,?,?,?)""", normalized_transitions)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {"experiences_imported":len(normalized_exp),"replay_transitions_imported":len(normalized_transitions),"format_version":1}

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
        if reward is not None and (
            isinstance(reward, bool) or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ValueError("reward must be a finite number or None")
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
            self._records_since_prune += 1
            if self._records_since_prune >= 1000:
                self._prune_unlinked_locked(exclude_id=entry["id"])
                self._records_since_prune = 0
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

    def remember_transition(
        self,
        *,
        agent: str,
        context: str,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        terminal: bool,
        experience_id: str | None = None,
        capacity: int = 10000,
    ) -> None:
        """Persist a transition in a bounded replay store for future DQN updates.

        Replay is partitioned by team and scenario (for example, 1v1 vs 3v2)
        so differently trained checkpoints do not silently mix their experiences.
        """
        if agent not in ("attacker", "defender"):
            raise ValueError("agent must be attacker or defender")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < 12:
            raise ValueError("action must be an integer from 0 through 11")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= 100000:
            raise ValueError("capacity must be between 1 and 100000")
        state_values = [float(value) for value in state]
        next_values = [float(value) for value in next_state]
        if not state_values or len(state_values) != len(next_values):
            raise ValueError("state and next_state must be non-empty and have equal dimensions")
        if not all(math.isfinite(value) for value in (*state_values, *next_values)):
            raise ValueError("state values must be finite numbers")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError("reward must be a finite number")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a boolean")
        entry = (
            datetime.now(timezone.utc).isoformat(), agent, context.strip(), experience_id,
            _json(state_values), action, float(reward), _json(next_values), int(terminal),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO replay_transitions
                   (created_at, agent, context, experience_id, state, action, reward,
                    next_state, terminal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                entry,
            )
            self._connection.execute(
                """DELETE FROM replay_transitions
                   WHERE agent = ? AND context = ? AND id NOT IN (
                       SELECT id FROM replay_transitions
                       WHERE agent = ? AND context = ?
                       ORDER BY id DESC LIMIT ?
                   )""",
                (agent, context.strip(), agent, context.strip(), capacity),
            )

    def load_transitions(
        self, *, agent: str, context: str, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Return a replay partition in chronological order for deterministic reload."""
        if agent not in ("attacker", "defender"):
            raise ValueError("agent must be attacker or defender")
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        with self._lock:
            rows = self._connection.execute(
                """SELECT state, action, reward, next_state, terminal
                   FROM replay_transitions
                   WHERE agent = ? AND context = ?
                   ORDER BY id DESC LIMIT ?""",
                (agent, context, limit),
            ).fetchall()
        rows.reverse()
        return [{
            "state": json.loads(row["state"]),
            "action": int(row["action"]),
            "reward": float(row["reward"]),
            "next_state": json.loads(row["next_state"]),
            "terminal": bool(row["terminal"]),
        } for row in rows]

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
        """Delete the audit record and any replay transition linked to it."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM experiences WHERE id = ?", (experience_id,)
            )
            return cursor.rowcount > 0

    def reset(self) -> int:
        """Delete all episode records and all replay transitions."""
        with self._lock, self._connection:
            transition_count = self._connection.execute(
                "SELECT COUNT(*) FROM replay_transitions"
            ).fetchone()[0]
            experience_count = self._connection.execute(
                "SELECT COUNT(*) FROM experiences"
            ).fetchone()[0]
            self._connection.execute("DELETE FROM replay_transitions")
            self._connection.execute("DELETE FROM experiences")
            return int(transition_count + experience_count)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """SELECT COUNT(*) AS total,
                   SUM(CASE WHEN outcome IN ('success', 'succeeded', 'completed') THEN 1 ELSE 0 END) AS successes,
                   SUM(CASE WHEN outcome IN ('failure', 'failed', 'error') THEN 1 ELSE 0 END) AS failures,
                   SUM(CASE WHEN lesson_status = 'active' THEN 1 ELSE 0 END) AS active_lessons
                   FROM experiences"""
            ).fetchone()
        with self._lock:
            replay_count = self._connection.execute(
                "SELECT COUNT(*) FROM replay_transitions"
            ).fetchone()[0]
            replay_by_agent = {
                item["agent"]: item["count"]
                for item in self._connection.execute(
                    "SELECT agent, COUNT(*) AS count FROM replay_transitions GROUP BY agent"
                ).fetchall()
            }
        total = row["total"] or 0
        successes = row["successes"] or 0
        failures = row["failures"] or 0
        return {
            "total_experiences": total,
            "successful_outcomes": successes,
            "failed_outcomes": failures,
            "active_lessons": row["active_lessons"] or 0,
            "persisted_replay_transitions": replay_count,
            "replay_transitions_by_agent": replay_by_agent,
            "observed_success_rate": successes / (successes + failures) if successes + failures else None,
            "note": "Observed outcome counts are not a controlled learning benchmark; replay transitions are used by resumed DQN training when loaded.",
        }
