"""Market-database resolver: real / synthetic / auto source selection.

Selects the SQLite database the framework reads, validates it against the
committed schema manifest (and optionally against a required date range),
and — in ``auto`` mode — creates or reuses the synthetic drop-in database
when the real one is unavailable. The selected source and its provenance
are logged loudly and exposed for experiment metadata.

The rest of the application accesses market data exclusively through
:class:`hybrid_vpp.data.sqlite_market_data.MarketDataStore`, which calls
this resolver — controllers, environments, and training code contain no
real-vs-synthetic branching.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import (
    DataConfig,
    MarketsConfig,
    SyntheticMarketConfig,
)
from hybrid_vpp.data.schema_manifest import validate_against_manifest

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarketDataSource:
    """The resolved database plus provenance for logs and run metadata."""

    path: Path
    provenance: str  # "real" | "synthetic" | "synthetic_calibrated"
    problems: tuple[str, ...] = ()  # why the real database was rejected (auto mode)
    metadata: dict = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return self.provenance.startswith("synthetic")


class MarketDatabaseError(RuntimeError):
    """Raised when the configured database cannot be used and no fallback applies."""


def _check_real(path: Path | None, required_period: tuple | None) -> list[str]:
    problems: list[str] = []
    if path is None:
        return [
            "missing_file: no real database configured "
            "(set $MARKET_DATABASE_PATH or data.market_database.path)"
        ]
    if not path.exists():
        return [f"missing_file: {path} does not exist"]
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            con.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.Error as exc:
        return [f"invalid_schema: not a readable SQLite database ({exc})"]
    problems += [f"invalid_schema: {p}" for p in validate_against_manifest(path)]
    if problems or required_period is None:
        return problems
    start, end = (pd.Timestamp(t) for t in required_period)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        lo, hi = con.execute('SELECT MIN("index"), MAX("index") FROM day_ahead_prices').fetchone()
        if lo is None or pd.Timestamp(lo) > start or pd.Timestamp(hi) < end - pd.Timedelta("2D"):
            problems.append(
                f"insufficient_coverage: day_ahead_prices covers [{lo} .. {hi}], "
                f"required [{start} .. {end})"
            )
    return problems


def _triggered(problems: list[str], data_cfg: DataConfig) -> bool:
    triggers = data_cfg.market_database.fallback_on
    mapping = {
        "missing_file": triggers.missing_file,
        "invalid_schema": triggers.invalid_schema,
        "insufficient_coverage": triggers.insufficient_coverage,
    }
    return all(mapping.get(p.split(":", 1)[0], False) for p in problems)


def resolve_market_database(
    data_cfg: DataConfig,
    markets_cfg: MarketsConfig,
    synthetic_cfg: SyntheticMarketConfig | None = None,
    required_period: tuple | None = None,
) -> MarketDataSource:
    """Resolve the active market database according to the configured mode."""
    from hybrid_vpp.data.synthetic_market import (
        generate_synthetic_database,
        read_metadata,
    )

    mdb = data_cfg.market_database
    synthetic_cfg = synthetic_cfg or SyntheticMarketConfig()
    real_path = mdb.resolved_real_path()

    def use_synthetic(problems: list[str]) -> MarketDataSource:
        if not mdb.synthetic_path.exists() and not mdb.create_if_missing:
            raise MarketDatabaseError(
                f"synthetic database {mdb.synthetic_path} missing and create_if_missing is false"
            )
        path = generate_synthetic_database(synthetic_cfg, markets_cfg, mdb.synthetic_path)
        metadata = read_metadata(path)
        provenance = (
            "synthetic_calibrated" if metadata.get("calibration") == "enabled" else "synthetic"
        )
        if problems:
            log.warning(
                "Real market database unavailable; using generated synthetic "
                "database: %s (reasons: %s)",
                path,
                "; ".join(problems),
            )
        else:
            log.warning("Using generated synthetic market database: %s", path)
        return MarketDataSource(path, provenance, tuple(problems), metadata)

    if mdb.mode == "synthetic":
        return use_synthetic([])

    problems = _check_real(real_path, required_period)
    if mdb.mode == "real":
        if problems:
            raise MarketDatabaseError(
                f"mode=real but {real_path} is unusable:\n  "
                + "\n  ".join(problems)
                + "\nSet $MARKET_DATABASE_PATH or data.market_database.path, "
                "or use mode=auto for the synthetic fallback."
            )
        log.info("Using real market database: %s", real_path)
        return MarketDataSource(real_path, "real")

    # mode == "auto"
    if not problems:
        log.info("Using real market database: %s", real_path)
        return MarketDataSource(real_path, "real")
    if _triggered(problems, data_cfg):
        return use_synthetic(problems)
    raise MarketDatabaseError(
        f"real database {real_path} unusable and fallback not permitted for:\n  "
        + "\n  ".join(problems)
    )


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module to see which source would be used
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")

if __name__ == "__main__":
    from hybrid_vpp.config.models import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(CONFIG_PATH)
    source = resolve_market_database(cfg.data, cfg.markets, cfg.synthetic_market)
    print(f"active source: {source.path} (provenance: {source.provenance})")
