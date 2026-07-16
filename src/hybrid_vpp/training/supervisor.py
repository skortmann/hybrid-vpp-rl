"""Autonomous research supervisor: reconcile → watchdog → schedule → report.

Runs a durable poll loop over the SQLite research registry:

1. acquire the PID-file lock (stale locks from dead supervisors are taken
   over automatically — restartable and idempotent);
2. reconcile registry rows against live processes and heartbeats;
3. apply stall criteria (heartbeat age, zero step progress, start timeout)
   and the bounded retry policy per failure class;
4. spawn fresh worker processes (one experiment per process, `spawn`
   semantics inside) while capacity is available and work is queued;
5. regenerate artifacts (screening table, progress report, next action)
   and append material events to the research ledger;
6. stop when the registry reaches a terminal state (or run one cycle).

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.training.supervisor
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from hybrid_vpp.training.research_state import (
    Heartbeat,
    ResearchRegistry,
    SupervisorLock,
    utcnow,
)

log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
LEDGER = Path("docs/rl_research_log.md")

#: anchors on the fixed 30 screening validation days (EUR/day), per economics
ANCHORS = {
    "do_nothing": 47908.0,
    "rule_based": 50861.0,
    "milp_info": 51165.0,
    "milp_perfect": 54994.0,
}
ANCHORS_V2_PATH = Path("artifacts/anchors_v2.json")


def anchors_for(phase: str) -> dict[str, float]:
    """env-v2 phases are gapped against the deviation-penalized anchors."""
    if phase.endswith("_v2") and ANCHORS_V2_PATH.exists():
        payload = json.loads(ANCHORS_V2_PATH.read_text())
        return {name: values["mean"] for name, values in payload["anchors"].items()}
    return ANCHORS


class Watchdog:
    def __init__(
        self,
        heartbeat_timeout_min: float = 6.0,
        no_progress_min: float = 15.0,
        start_timeout_min: float = 6.0,
        eval_timeout_min: float = 30.0,
    ) -> None:
        self.heartbeat_timeout_s = heartbeat_timeout_min * 60
        self.no_progress_s = no_progress_min * 60
        self.start_timeout_s = start_timeout_min * 60
        self.eval_timeout_s = eval_timeout_min * 60

    @staticmethod
    def pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def diagnose(self, row: dict, now: float) -> str | None:
        """Return a stall reason, or None when the job looks healthy."""
        started = (
            datetime.fromisoformat(row["started_at"]).timestamp() if row["started_at"] else now
        )
        if not self.pid_alive(row["pid"]):
            return "process dead without terminal state"

        heartbeat = Heartbeat.read(Path(row["heartbeat_path"])) if row["heartbeat_path"] else None
        if heartbeat is None:
            if now - started > self.start_timeout_s:
                return "no heartbeat after start timeout"
            return None
        hb_age = now - datetime.fromisoformat(heartbeat.timestamp).timestamp()
        if heartbeat.phase == "evaluating":
            if hb_age > self.eval_timeout_s:
                return f"evaluation heartbeat stale for {hb_age / 60:.0f} min"
            return None
        if hb_age > self.heartbeat_timeout_s:
            return f"heartbeat stale for {hb_age / 60:.0f} min"
        # step progress: compare with last registry snapshot
        if (
            heartbeat.environment_steps >= 0
            and heartbeat.environment_steps == (row["last_env_steps"] or 0)
            and row["updated_at"]
            and now - datetime.fromisoformat(row["updated_at"]).timestamp() > self.no_progress_s
        ):
            return (
                f"no environment-step progress for {self.no_progress_s / 60:.0f} min "
                f"(stuck at {heartbeat.environment_steps})"
            )
        return None


def kill_tree(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class Supervisor:
    def __init__(
        self,
        registry_path: Path = Path("runs/research_state.sqlite"),
        max_parallel: int = 3,
        poll_seconds: float = 60.0,
    ) -> None:
        self.registry = ResearchRegistry(registry_path)
        self.registry_path = registry_path
        self.max_parallel = max_parallel
        self.poll_seconds = poll_seconds
        self.watchdog = Watchdog()
        self.lock = SupervisorLock(registry_path.parent / "supervisor.lock")
        # ledger cursor persists across supervisor restarts (no duplicate entries)
        self._cursor_path = registry_path.parent / "ledger.cursor"
        try:
            self._ledger_event_id = int(self._cursor_path.read_text().strip())
        except (OSError, ValueError):
            self._ledger_event_id = 0

    # ------------------------------------------------------------- reconcile

    def reconcile(self) -> None:
        now = time.time()
        for row in self.registry.active():
            reason = self.watchdog.diagnose(row, now)
            heartbeat = (
                Heartbeat.read(Path(row["heartbeat_path"])) if row["heartbeat_path"] else None
            )
            if heartbeat is not None and heartbeat.environment_steps != (
                row["last_env_steps"] or 0
            ):
                self.registry.update(
                    row["experiment_id"],
                    last_env_steps=heartbeat.environment_steps,
                    last_heartbeat=heartbeat.timestamp,
                )
            if reason is None:
                continue
            experiment_id = row["experiment_id"]
            log.warning("stall detected for %s: %s", experiment_id, reason)
            if row["pid"]:
                kill_tree(row["pid"])
            tail = self._log_tail(row)
            self.registry.update(
                experiment_id,
                state="STALE",
                failure_class="PROCESS_DEADLOCK",
                failure_detail=f"{reason}\nlog tail:\n{tail}",
            )
            self.registry.log_event(experiment_id, "stalled", reason)
            self._apply_retry_policy(self.registry.get(experiment_id))

    def _log_tail(self, row: dict, lines: int = 20) -> str:
        logs = sorted(
            Path("runs/worker_logs").glob("worker_*.log"), key=lambda p: p.stat().st_mtime
        )
        for log_path in reversed(logs):  # newest log mentioning this experiment
            text = log_path.read_text()
            if row["experiment_id"] in text:
                return "\n".join(text.splitlines()[-lines:])
        return "(no log)"

    def _apply_retry_policy(self, row: dict) -> None:
        experiment_id = row["experiment_id"]
        if row["retry_count"] >= row["max_retries"]:
            self.registry.update(experiment_id, state="FAILED_FINAL")
            self.registry.log_event(
                experiment_id,
                "failed_final",
                f"retries exhausted ({row['retry_count']}/{row['max_retries']}), "
                f"class={row['failure_class']}",
            )
            return
        self.registry.update(
            experiment_id,
            state="QUEUED",
            retry_count=row["retry_count"] + 1,
            pid=None,
        )
        self.registry.log_event(
            experiment_id,
            "requeued",
            f"retry {row['retry_count'] + 1}/{row['max_retries']} after {row['failure_class']}",
        )

    def handle_failed_retryable(self) -> None:
        for row in self.registry.all(state="FAILED_RETRYABLE"):
            self._apply_retry_policy(row)

    # -------------------------------------------------------------- schedule

    def schedule(self) -> None:
        active = self.registry.active()
        capacity = self.max_parallel - len(active)
        queued = self.registry.all(state="QUEUED")
        for _ in range(min(capacity, len(queued))):
            self._spawn_worker()

    def _spawn_worker(self) -> None:
        logs = Path("runs/worker_logs")
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%H%M%S%f")
        out = (logs / f"worker_{stamp}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, "-m", "hybrid_vpp.training.worker"],
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group => clean kill_tree
            cwd=str(Path.cwd()),
        )
        self.registry.log_event(None, "worker_spawned", f"pid={process.pid}")

    # --------------------------------------------------------------- reports

    def regenerate_artifacts(self) -> None:
        ARTIFACTS.mkdir(exist_ok=True)
        rows = self.registry.all()
        completed = [
            {**r, "result": json.loads(r["result_json"])}
            for r in rows
            if r["result_json"] and r["state"] in ("COMPLETED", "PROMOTED")
        ]
        completed.sort(key=lambda r: -r["result"]["mean_revenue_eur"])

        lines = [
            "| experiment | phase | state | mean EUR/day | median | info-gap |",
            "|---|---|---|---|---|---|",
        ]
        csv = ["experiment_id,phase,state,mean_eur_day,median_eur_day,info_gap_pct"]
        for r in completed:
            mean = r["result"]["mean_revenue_eur"]
            median = r["result"].get("median_revenue_eur", float("nan"))
            anchors = anchors_for(r["phase"])
            gap = (anchors["milp_info"] - mean) / anchors["milp_info"] * 100
            gap_str = f"{gap:+.1f}%" if r["kind"] == "training" else "n/a"
            lines.append(
                f"| {r['experiment_id']} | {r['phase']} | {r['state']} "
                f"| {mean:,.0f} | {median:,.0f} | {gap_str} |"
            )
            csv.append(
                f"{r['experiment_id']},{r['phase']},{r['state']},{mean:.0f},{median:.0f},{gap:.2f}"
            )
        (ARTIFACTS / "screening_table.md").write_text("\n".join(lines) + "\n")
        (ARTIFACTS / "screening_table.csv").write_text("\n".join(csv) + "\n")

        best = completed[0] if completed else None
        if best and best["kind"] == "training":
            (ARTIFACTS / "current_best.json").write_text(
                json.dumps(
                    {
                        "experiment_id": best["experiment_id"],
                        "mean_revenue_eur": best["result"]["mean_revenue_eur"],
                        "median_revenue_eur": best["result"].get("median_revenue_eur"),
                        "anchors": ANCHORS,
                        "generated_at": utcnow(),
                    },
                    indent=2,
                )
            )

        counts = self.registry.counts()
        terminal, summary = self.registry.is_terminal()
        progress = [
            f"# Research progress ({utcnow()})",
            "",
            f"State counts: `{json.dumps(counts)}`",
            f"Terminal: **{terminal}**",
            "",
            "Anchors (30 fixed validation days, EUR/day): "
            + ", ".join(f"{k} {v:,.0f}" for k, v in ANCHORS.items()),
            "",
            "See `screening_table.md` for ranked results and "
            "`docs/rl_research_log.md` for the event ledger.",
        ]
        (ARTIFACTS / "research_progress.md").write_text("\n".join(progress) + "\n")

        queued = self.registry.all(state="QUEUED")
        next_action = {
            "generated_at": utcnow(),
            "terminal": terminal,
            "state_counts": counts,
            "next_action": (
                f"run {queued[0]['experiment_id']}"
                if queued
                else ("terminal" if terminal else "await active jobs")
            ),
            "reason": summary,
            "command": "uv run python -m hybrid_vpp.training.supervisor",
            "verification": [
                "heartbeat advances",
                "checkpoint reloads",
                "result validated against invariants",
            ],
        }
        (ARTIFACTS / "next_action.json").write_text(json.dumps(next_action, indent=2))

    def append_ledger(self) -> None:
        events = self.registry.events_since(self._ledger_event_id)
        if not events:
            return
        LEDGER.parent.mkdir(exist_ok=True)
        if not LEDGER.exists():
            LEDGER.write_text(
                "# RL research ledger\n\nAppend-only event log (generated by the supervisor).\n\n"
            )
        with LEDGER.open("a") as fh:
            for e in events:
                fh.write(
                    f"- `{e['ts']}` **{e['event']}** {e['experiment_id'] or ''} — {e['detail']}\n"
                )
        self._ledger_event_id = events[-1]["id"]
        self._cursor_path.write_text(str(self._ledger_event_id))

    # ------------------------------------------------------------------ run

    def cycle(self) -> bool:
        """One supervision cycle; returns True when terminal."""
        self.reconcile()
        self.handle_failed_retryable()
        self.schedule()
        self.regenerate_artifacts()
        self.append_ledger()
        terminal, _ = self.registry.is_terminal()
        return terminal

    def run(self, until_terminal: bool = True, max_cycles: int | None = None) -> None:
        if not self.lock.acquire():
            log.error("another supervisor (pid %s) holds the lock", self.lock.holder())
            return
        try:
            cycles = 0
            while True:
                terminal = self.cycle()
                cycles += 1
                if terminal and until_terminal:
                    self.registry.log_event(None, "terminal_state_reached", "")
                    self.append_ledger()
                    log.info("terminal state reached after %d cycles", cycles)
                    return
                if max_cycles is not None and cycles >= max_cycles:
                    return
                if not until_terminal:
                    return
                time.sleep(self.poll_seconds)
        finally:
            self.lock.release()


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
REGISTRY_PATH = Path("runs/research_state.sqlite")
MAX_PARALLEL = 3
POLL_SECONDS = 60.0
UNTIL_TERMINAL = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    Supervisor(REGISTRY_PATH, MAX_PARALLEL, POLL_SECONDS).run(UNTIL_TERMINAL)
