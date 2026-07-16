"""Persistent research-state registry (SQLite) for the autonomous RL loop.

One row per experiment (or analysis job). Workers *claim* the highest-
priority ``QUEUED`` row atomically, so any number of spawned workers can
cooperate without a central dispatcher handing out IDs; the supervisor
only tops up workers, reconciles health, and applies retry policy.

States (§research protocol): PLANNED → QUEUED → STARTING → RUNNING →
EVALUATING → COMPLETED | INVALID_RESULT | STALE | FAILED_RETRYABLE |
FAILED_FINAL | PRUNED | PROMOTED | SUPERSEDED | BLOCKED_EXTERNAL.

Every state change is appended to an ``events`` table with a reason —
the human-readable ledger and ``next_action.json`` are generated from it.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

STATES = (
    "PLANNED",
    "QUEUED",
    "STARTING",
    "RUNNING",
    "EVALUATING",
    "VALIDATING",
    "COMPLETED",
    "PROMOTED",
    "PRUNED",
    "FAILED_RETRYABLE",
    "FAILED_FINAL",
    "STALE",
    "BLOCKED_EXTERNAL",
    "INVALID_RESULT",
    "SUPERSEDED",
)

TERMINAL_STATES = {
    "COMPLETED",
    "PROMOTED",
    "PRUNED",
    "FAILED_FINAL",
    "INVALID_RESULT",
    "SUPERSEDED",
}
ACTIVE_STATES = {"STARTING", "RUNNING", "EVALUATING", "VALIDATING"}

FAILURE_CLASSES = (
    "PROCESS_DEADLOCK",
    "SUBPROCESS_START_FAILURE",
    "OUT_OF_MEMORY",
    "NAN_OR_INF",
    "ENVIRONMENT_EXCEPTION",
    "ACCOUNTING_INVARIANT_FAILURE",
    "CHECKPOINT_CORRUPTION",
    "WALL_CLOCK_TIMEOUT",
    "WANDB_FAILURE",
    "TOOL_PERMISSION_FAILURE",
    "DATABASE_FAILURE",
    "INVALID_CONFIGURATION",
    "NO_LEARNING",
    "STATISTICALLY_DOMINATED",
    "UNKNOWN",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    kind          TEXT NOT NULL DEFAULT 'training',
    phase         TEXT NOT NULL,
    spec_json     TEXT NOT NULL,
    priority      INTEGER NOT NULL DEFAULT 50,
    state         TEXT NOT NULL DEFAULT 'PLANNED',
    retry_count   INTEGER NOT NULL DEFAULT 0,
    max_retries   INTEGER NOT NULL DEFAULT 2,
    pid           INTEGER,
    hostname      TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    last_heartbeat TEXT,
    last_env_steps INTEGER DEFAULT 0,
    heartbeat_path TEXT,
    run_dir       TEXT,
    wandb_run_id  TEXT,
    failure_class TEXT,
    failure_detail TEXT,
    result_json   TEXT,
    decision      TEXT,
    decision_reason TEXT,
    git_commit    TEXT,
    schema_versions TEXT,
    updated_at    TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    experiment_id TEXT,
    event TEXT NOT NULL,
    detail TEXT
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Heartbeat:
    experiment_id: str
    timestamp: str
    pid: int
    environment_steps: int
    phase: str
    latest_validation_return: float | None = None
    checkpoint: str | None = None
    wandb_run_id: str | None = None

    def write(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)  # atomic

    @staticmethod
    def read(path: Path) -> Heartbeat | None:
        try:
            return Heartbeat(**json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            return None


class ResearchRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    # -------------------------------------------------------------- writes

    def add(
        self,
        experiment_id: str,
        spec: dict,
        phase: str,
        priority: int = 50,
        kind: str = "training",
        state: str = "QUEUED",
        max_retries: int = 2,
    ) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state}")
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiments "
                "(experiment_id, kind, phase, spec_json, priority, state, max_retries,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    kind,
                    phase,
                    json.dumps(spec),
                    priority,
                    state,
                    max_retries,
                    utcnow(),
                ),
            )
        self.log_event(experiment_id, "added", f"phase={phase} state={state}")

    def update(self, experiment_id: str, **fields) -> None:
        if "state" in fields and fields["state"] not in STATES:
            raise ValueError(f"unknown state {fields['state']}")
        fields["updated_at"] = utcnow()
        keys = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as con:
            con.execute(
                f"UPDATE experiments SET {keys} WHERE experiment_id = ?",  # noqa: S608
                (*fields.values(), experiment_id),
            )

    def claim_next(self, pid: int) -> dict | None:
        """Atomically claim the highest-priority QUEUED row (worker side)."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT experiment_id FROM experiments WHERE state = 'QUEUED' "
                "ORDER BY priority DESC, experiment_id LIMIT 1"
            ).fetchone()
            if row is None:
                con.execute("COMMIT")
                return None
            con.execute(
                "UPDATE experiments SET state='STARTING', pid=?, hostname=?, "
                "started_at=?, updated_at=? WHERE experiment_id=? AND state='QUEUED'",
                (pid, socket.gethostname(), utcnow(), utcnow(), row["experiment_id"]),
            )
            con.execute("COMMIT")
        self.log_event(row["experiment_id"], "claimed", f"pid={pid}")
        return self.get(row["experiment_id"])

    def log_event(self, experiment_id: str | None, event: str, detail: str = "") -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO events (ts, experiment_id, event, detail) VALUES (?,?,?,?)",
                (utcnow(), experiment_id, event, detail),
            )

    # --------------------------------------------------------------- reads

    def get(self, experiment_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return dict(row) if row else None

    def all(self, state: str | None = None) -> list[dict]:
        query = "SELECT * FROM experiments"
        args: tuple = ()
        if state:
            query += " WHERE state = ?"
            args = (state,)
        with self._connect() as con:
            return [dict(r) for r in con.execute(query + " ORDER BY priority DESC", args)]

    def counts(self) -> dict[str, int]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT state, COUNT(*) n FROM experiments GROUP BY state"
            ).fetchall()
        return {r["state"]: r["n"] for r in rows}

    def events_since(self, event_id: int = 0) -> list[dict]:
        with self._connect() as con:
            return [
                dict(r)
                for r in con.execute("SELECT * FROM events WHERE id > ? ORDER BY id", (event_id,))
            ]

    def active(self) -> list[dict]:
        with self._connect() as con:
            marks = ",".join(f"'{s}'" for s in ACTIVE_STATES)
            return [
                dict(r)
                for r in con.execute(
                    f"SELECT * FROM experiments WHERE state IN ({marks})"  # noqa: S608
                )
            ]

    def is_terminal(self) -> tuple[bool, str]:
        """(all jobs terminal?, summary)."""
        counts = self.counts()
        open_states = {s: n for s, n in counts.items() if s not in TERMINAL_STATES and n > 0}
        return (not open_states, json.dumps({"open": open_states, "all": counts}))


class SupervisorLock:
    """PID-file lock with stale-lock takeover (dead pid => acquirable)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)  # raises if dead
                return pid == os.getpid()
            except (ValueError, ProcessLookupError):
                self.path.unlink(missing_ok=True)  # stale lock
            except PermissionError:
                return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}")
        tmp.write_text(str(os.getpid()))
        try:
            os.link(tmp, self.path)  # atomic acquire
            return True
        except FileExistsError:
            return int(self.path.read_text().strip()) == os.getpid()
        finally:
            tmp.unlink(missing_ok=True)

    def release(self) -> None:
        try:
            if int(self.path.read_text().strip()) == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass

    def holder(self) -> int | None:
        try:
            pid = int(self.path.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError):
            return None


class HeartbeatThread:
    """Background heartbeat emitter for long non-training phases (dataset
    builds, evaluations) where no SB3 callback is running."""

    def __init__(
        self, path: Path, experiment_id: str, phase: str, interval_s: float = 30.0
    ) -> None:
        import threading

        self.path = path
        self.experiment_id = experiment_id
        self.phase = phase
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            Heartbeat(
                experiment_id=self.experiment_id,
                timestamp=utcnow(),
                pid=os.getpid(),
                environment_steps=-1,
                phase=self.phase,
            ).write(self.path)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> HeartbeatThread:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def wait_and_retry_claim(registry: ResearchRegistry, attempts: int = 3) -> dict | None:
    for k in range(attempts):
        try:
            return registry.claim_next(os.getpid())
        except sqlite3.OperationalError:
            time.sleep(1.0 + k)
    return None
