# Contributing

## Local setup

```bash
git clone https://github.com/skortmann/hybrid-vpp-rl
cd hybrid-vpp-rl
uv sync --group dev
cp .env.example local.env         # fill in tokens/paths as needed (never commit)
uv run python -m hybrid_vpp.data.synthetic_market   # offline market data
uv run pytest                     # 107 offline tests; +22 with a real database
```

Set `MARKET_DATABASE_PATH` in `local.env` to run the real-database
integration tests; without it they are skipped automatically.

## Quality gates

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv run mkdocs build
```

CI runs the same checks (offline tests only). Please keep new code within
these gates; do not weaken a check to make it pass.

## Conventions

* Conventional-commit messages: `feat: ...`, `fix: ...`, `test: ...`,
  `docs: ...`, `chore: ...`, `ci: ...`.
* All internal timestamps are tz-aware UTC; market wall-clock times exist
  only in the market calendar. Every power↔energy conversion goes through
  `core.timegrid.energy_mwh`.
* Configuration lives in pydantic models (`config/models.py`) + YAML — no
  magic numbers inside environment or controller code.
* Runnable modules use a marked `CONFIG` block of module constants instead
  of CLI flags; expose a plain function with keyword arguments so logic can
  be imported.
* Tests accompany every behavioral change; accounting and market-timing
  claims need hand-calculated cases.

## Extending the framework

* **New market / auction session**: add an `AuctionSessionConfig` (or a new
  config model) in `config/models.py`, generate its events and product
  eligibility in `markets/calendar.py`, and an execution model in
  `markets/execution.py`. Gate/eligibility rules belong in the calendar —
  never inside controllers.
* **New asset model**: implement it under `assets/` with explicit feasible
  bounds and an audit record for every correction (see `assets/battery.py`),
  then wire it into the feasibility projection and `sim/simulator.py`.
* **New controller**: implement the `Controller` protocol
  (`controllers/base.py`) — one `act(event, sim)` per decision event; add it
  to `training/evaluate.py` for benchmark comparison.
* **New dataset adapter**: implement the series accessors of
  `data/sqlite_market_data.py` against your source, or extend the schema
  manifest + synthetic generator pair so the drop-in contract stays intact
  (`docs/synthetic_market_data.md`).
* **New forecast provider**: implement `forecast(issue_time, delivery_times)`
  (`forecasts/base.py`); providers must never expose information published
  after the issue time — add a leakage test under `tests/leakage/`.
