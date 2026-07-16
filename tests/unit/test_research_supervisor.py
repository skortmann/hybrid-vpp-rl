"""Autonomous research harness: registry, watchdog, retries, validation."""

import json
import os
import time

import numpy as np
import pytest

from hybrid_vpp.training.research_state import (
    Heartbeat,
    ResearchRegistry,
    SupervisorLock,
    utcnow,
)
from hybrid_vpp.training.supervisor import Supervisor, Watchdog
from hybrid_vpp.training.worker import validate_training_result


@pytest.fixture()
def registry(tmp_path) -> ResearchRegistry:
    return ResearchRegistry(tmp_path / "state.sqlite")


def add_job(registry, experiment_id="exp-1", priority=50, max_retries=2):
    registry.add(
        experiment_id,
        {"experiment_id": experiment_id, "phase": "screening"},
        phase="screening",
        priority=priority,
        max_retries=max_retries,
    )


class TestRegistry:
    def test_claim_is_atomic_and_priority_ordered(self, registry):
        add_job(registry, "low", priority=10)
        add_job(registry, "high", priority=90)
        first = registry.claim_next(pid=111)
        second = registry.claim_next(pid=222)
        assert first["experiment_id"] == "high"
        assert second["experiment_id"] == "low"
        assert registry.claim_next(pid=333) is None  # nothing left: no duplicates

    def test_no_duplicate_launch_after_claim(self, registry):
        add_job(registry, "only")
        assert registry.claim_next(pid=1)["experiment_id"] == "only"
        assert registry.claim_next(pid=2) is None
        assert registry.get("only")["state"] == "STARTING"

    def test_invalid_state_rejected(self, registry):
        add_job(registry)
        with pytest.raises(ValueError):
            registry.update("exp-1", state="RUNNING_MAYBE")

    def test_terminal_detection(self, registry):
        add_job(registry, "a")
        add_job(registry, "b")
        assert registry.is_terminal()[0] is False
        registry.update("a", state="COMPLETED")
        registry.update("b", state="FAILED_FINAL")
        assert registry.is_terminal()[0] is True


class TestWatchdog:
    def make_row(
        self,
        tmp_path,
        *,
        pid,
        heartbeat_age_s=None,
        steps=100,
        phase="training",
        started_ago_s=600,
        last_env_steps=0,
        updated_ago_s=0,
    ):
        heartbeat_path = None
        if heartbeat_age_s is not None:
            heartbeat_path = tmp_path / "hb.json"
            hb = Heartbeat("exp", utcnow(), pid, steps, phase)
            hb.write(heartbeat_path)
            stamp = time.time() - heartbeat_age_s
            ts = (
                __import__("datetime")
                .datetime.fromtimestamp(stamp, __import__("datetime").UTC)
                .isoformat()
            )
            data = json.loads(heartbeat_path.read_text())
            data["timestamp"] = ts
            heartbeat_path.write_text(json.dumps(data))
        now = time.time()

        def iso(seconds_ago: float) -> str:
            import datetime

            return datetime.datetime.fromtimestamp(now - seconds_ago, datetime.UTC).isoformat()
        return {
            "experiment_id": "exp",
            "pid": pid,
            "heartbeat_path": str(heartbeat_path) if heartbeat_path else None,
            "started_at": iso(started_ago_s),
            "updated_at": iso(updated_ago_s),
            "last_env_steps": last_env_steps,
        }

    def test_dead_pid_is_stalled(self, tmp_path):
        row = self.make_row(tmp_path, pid=999999983, heartbeat_age_s=10)
        assert "dead" in Watchdog().diagnose(row, time.time())

    def test_fresh_heartbeat_is_healthy(self, tmp_path):
        row = self.make_row(
            tmp_path, pid=os.getpid(), heartbeat_age_s=5, steps=100, last_env_steps=50
        )
        assert Watchdog().diagnose(row, time.time()) is None

    def test_stale_heartbeat_detected(self, tmp_path):
        row = self.make_row(tmp_path, pid=os.getpid(), heartbeat_age_s=999)
        assert "stale" in Watchdog(heartbeat_timeout_min=5).diagnose(row, time.time())

    def test_no_step_progress_detected(self, tmp_path):
        row = self.make_row(
            tmp_path,
            pid=os.getpid(),
            heartbeat_age_s=5,
            steps=100,
            last_env_steps=100,
            updated_ago_s=999,
        )
        reason = Watchdog(no_progress_min=10).diagnose(row, time.time())
        assert reason and "progress" in reason

    def test_missing_heartbeat_within_grace_is_ok(self, tmp_path):
        row = self.make_row(tmp_path, pid=os.getpid(), heartbeat_age_s=None, started_ago_s=30)
        assert Watchdog().diagnose(row, time.time()) is None


class TestRetryPolicy:
    def test_bounded_retries_then_failed_final(self, tmp_path):
        supervisor = Supervisor(tmp_path / "s.sqlite", max_parallel=0)
        registry = supervisor.registry
        add_job(registry, "flaky", max_retries=2)
        registry.claim_next(pid=os.getpid())
        for expected_state in ("QUEUED", "QUEUED", "FAILED_FINAL"):
            registry.update("flaky", state="FAILED_RETRYABLE", failure_class="PROCESS_DEADLOCK")
            supervisor.handle_failed_retryable()
            assert registry.get("flaky")["state"] == expected_state
            if expected_state == "QUEUED":
                registry.claim_next(pid=os.getpid())

    def test_supervisor_cycle_reaches_terminal(self, tmp_path):
        supervisor = Supervisor(tmp_path / "s.sqlite", max_parallel=0)
        add_job(supervisor.registry, "done-job")
        supervisor.registry.update(
            "done-job",
            state="COMPLETED",
            result_json=json.dumps(
                {"mean_revenue_eur": 1.0, "median_revenue_eur": 1.0, "per_day_revenue_eur": [1.0]}
            ),
        )
        os.chdir(tmp_path)  # artifacts written into tmp
        assert supervisor.cycle() is True
        assert (tmp_path / "artifacts/next_action.json").exists()
        payload = json.loads((tmp_path / "artifacts/next_action.json").read_text())
        assert payload["terminal"] is True


class TestSupervisorLock:
    def test_acquire_and_duplicate_prevention(self, tmp_path):
        lock = SupervisorLock(tmp_path / "lock")
        assert lock.acquire() is True
        assert lock.acquire() is True  # reentrant for same pid

    def test_stale_lock_takeover(self, tmp_path):
        path = tmp_path / "lock"
        path.write_text("999999983")  # dead pid
        lock = SupervisorLock(path)
        assert lock.acquire() is True
        assert int(path.read_text()) == os.getpid()


class TestResultValidation:
    def test_rejects_missing_checkpoint(self, tmp_path):
        result = {"mean_revenue_eur": 1.0, "per_day_revenue_eur": [1.0] * 30}
        problems = validate_training_result(result, tmp_path / "c.yaml", tmp_path / "missing.zip")
        assert any("checkpoint" in p for p in problems)

    def test_rejects_nonfinite_and_wrong_day_count(self, tmp_path):
        result = {"mean_revenue_eur": np.nan, "per_day_revenue_eur": [np.nan] * 10}
        problems = validate_training_result(result, tmp_path / "c.yaml", tmp_path / "missing.zip")
        assert any("30" in p for p in problems)
        assert any("finite" in p for p in problems)

    def test_rejects_inconsistent_aggregate(self, tmp_path):
        result = {"mean_revenue_eur": 999.0, "per_day_revenue_eur": [1.0] * 30}
        problems = validate_training_result(result, tmp_path / "c.yaml", tmp_path / "missing.zip")
        assert any("aggregate" in p for p in problems)
