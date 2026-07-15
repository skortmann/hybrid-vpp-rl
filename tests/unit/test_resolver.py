"""Market-database resolver: mode semantics, fallback triggers, provenance."""

import sqlite3
from pathlib import Path

import pytest

from hybrid_vpp.config.models import (
    DataConfig,
    FallbackTriggers,
    MarketDatabaseConfig,
    MarketsConfig,
    SyntheticMarketConfig,
)
from hybrid_vpp.data.resolver import MarketDatabaseError, resolve_market_database
from hybrid_vpp.data.synthetic_market import generate_synthetic_database

MARKETS = MarketsConfig()
SMALL = SyntheticMarketConfig(start="2025-02-01", end="2025-02-08", random_seed=3)


def data_cfg(tmp_path: Path, mode: str, real: Path | None = None, **fallback) -> DataConfig:
    return DataConfig(
        market_database=MarketDatabaseConfig(
            mode=mode,
            path=real,
            synthetic_path=tmp_path / "synthetic.db",
            fallback_on=FallbackTriggers(**fallback),
        ),
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture(scope="module")
def valid_real(tmp_path_factory) -> Path:
    """A schema-valid stand-in for the real database (a generated one)."""
    path = tmp_path_factory.mktemp("real") / "real.db"
    return generate_synthetic_database(SMALL, MARKETS, path)


def test_real_mode_uses_valid_database(tmp_path, valid_real):
    source = resolve_market_database(data_cfg(tmp_path, "real", real=valid_real), MARKETS, SMALL)
    assert source.path == valid_real
    assert source.provenance == "real"


def test_real_mode_fails_clearly_when_missing(tmp_path):
    cfg = data_cfg(tmp_path, "real", real=tmp_path / "nope.db")
    with pytest.raises(MarketDatabaseError, match="missing_file"):
        resolve_market_database(cfg, MARKETS, SMALL)
    assert not (tmp_path / "synthetic.db").exists()  # never silently synthetic


def test_synthetic_mode_ignores_real(tmp_path, valid_real):
    cfg = data_cfg(tmp_path, "synthetic", real=valid_real)
    source = resolve_market_database(cfg, MARKETS, SMALL)
    assert source.path == tmp_path / "synthetic.db"
    assert source.is_synthetic
    assert source.metadata["synthetic"] == "true"


def test_auto_prefers_valid_real(tmp_path, valid_real):
    source = resolve_market_database(data_cfg(tmp_path, "auto", real=valid_real), MARKETS, SMALL)
    assert source.provenance == "real"


def test_auto_falls_back_when_missing(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = data_cfg(tmp_path, "auto", real=tmp_path / "gone.db")
    source = resolve_market_database(cfg, MARKETS, SMALL)
    assert source.is_synthetic
    assert any("synthetic" in r.message.lower() for r in caplog.records)
    assert source.problems and source.problems[0].startswith("missing_file")


def test_auto_falls_back_on_invalid_schema(tmp_path):
    bogus = tmp_path / "bogus.db"
    with sqlite3.connect(bogus) as con:
        con.execute("CREATE TABLE unrelated (x INT)")
    cfg = data_cfg(tmp_path, "auto", real=bogus)
    source = resolve_market_database(cfg, MARKETS, SMALL)
    assert source.is_synthetic
    assert any(p.startswith("invalid_schema") for p in source.problems)


def test_auto_respects_disabled_fallback(tmp_path):
    cfg = data_cfg(tmp_path, "auto", real=tmp_path / "gone.db", missing_file=False)
    with pytest.raises(MarketDatabaseError, match="fallback not permitted"):
        resolve_market_database(cfg, MARKETS, SMALL)


def test_auto_falls_back_on_insufficient_coverage(tmp_path, valid_real):
    cfg = data_cfg(tmp_path, "auto", real=valid_real)
    source = resolve_market_database(
        cfg, MARKETS, SMALL, required_period=("2030-01-01", "2030-06-01")
    )
    assert source.is_synthetic
    assert any(p.startswith("insufficient_coverage") for p in source.problems)


def test_create_if_missing_false_raises(tmp_path):
    cfg = data_cfg(tmp_path, "synthetic")
    cfg.market_database.create_if_missing = False
    with pytest.raises(MarketDatabaseError, match="create_if_missing"):
        resolve_market_database(cfg, MARKETS, SMALL)
