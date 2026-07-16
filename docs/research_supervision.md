# Autonomous research supervision

The RL research program runs under a durable supervision loop instead of
ad-hoc shell jobs:

```
registry (runs/research_state.sqlite)
      ↓ claim (atomic)
worker process (one experiment, spawn-isolated)  →  heartbeat (30 s)
      ↓ result + DONE (atomic)
supervisor (poll loop)  →  watchdog / retries / scheduling
      ↓
artifacts/*.md|json + docs/rl_research_log.md
```

## Components

* **Registry** (`training/research_state.py`): SQLite (WAL) with one row
  per experiment; explicit states (`QUEUED → STARTING → RUNNING →
  EVALUATING → VALIDATING → COMPLETED | STALE | FAILED_* | ...`), retry
  counters, heartbeat pointers, results, decisions, and an append-only
  events table. Workers claim jobs atomically — duplicate launches are
  impossible by construction.
* **Worker** (`training/worker.py`): fresh process per experiment (the
  supervisor never reuses a process that has already trained — the
  fork-after-threads deadlock class is eliminated together with `spawn`
  vectorized environments). Writes heartbeats, evaluates on the fixed
  validation days, revalidates its own artifacts (checkpoint reload,
  finite values, aggregate recompute), writes `DONE` atomically, records
  the result, exits. W&B failures trigger one offline-mode retry — valid
  training is never discarded because remote logging failed.
* **Supervisor** (`training/supervisor.py`): lock-protected poll loop —
  reconcile registry vs. live PIDs and heartbeats, detect stalls
  (heartbeat age, zero step progress, start/eval timeouts), kill stalled
  process trees, classify failures, apply bounded retries
  (`max_retries`, default 2; scientific changes are new experiments, not
  retries), top up workers (`MAX_PARALLEL`), regenerate
  `artifacts/screening_table.{md,csv}`, `current_best.json`,
  `research_progress.md`, `next_action.json`, and append every material
  event to `docs/rl_research_log.md`. Idempotent and restartable; stale
  locks from dead supervisors are taken over automatically.

## Running

```bash
uv run python -m hybrid_vpp.training.supervisor   # until terminal state
```

For a machine-persistent setup, install the user service template (do not
commit machine-specific activation):

```bash
cp deploy/hybrid-vpp-rl-supervisor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hybrid-vpp-rl-supervisor
```

## Terminal states

The loop stops only in a named terminal state — `SUCCESS_NEAR_OPTIMAL`,
`SUCCESS_ABOVE_INFO_ANCHOR`, `EMPIRICAL_FRONTIER_REACHED`,
`BLOCKED_EXTERNAL_FINAL`, or `FAILED_SAFETY_OR_VALIDITY` — with the
machine-verifiable evidence recorded in `artifacts/` (see
`docs/rl_research.md` for the scientific criteria).
