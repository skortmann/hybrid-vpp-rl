"""Shared test configuration.

Tests that need the real (private) market database are skipped unless
``MARKET_DATABASE_PATH`` points at an existing file — the default suite
runs fully offline (synthetic fixtures are generated on the fly).
``local.env`` / ``.env`` are honored so local development picks up the
configured database automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hybrid_vpp.config.models import MARKET_DB_ENV_VAR, _load_env_files

_load_env_files()

_env = os.environ.get(MARKET_DB_ENV_VAR, "").strip()
REAL_DB_PATH: Path | None = Path(_env) if _env else None
REAL_DB_AVAILABLE = REAL_DB_PATH is not None and REAL_DB_PATH.exists()

requires_real_db = pytest.mark.skipif(
    not REAL_DB_AVAILABLE,
    reason="real market database not available (set MARKET_DATABASE_PATH)",
)
