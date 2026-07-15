"""Renewables.ninja download client for site-level wind and PV profiles.

The API token is read from the environment variable
``RENEWABLES_NINJA_TOKEN`` and is never written to disk or committed.

Every completed request is cached: the raw JSON response, a normalized
Parquet profile, and request metadata are stored under the output
directory. Cached year-chunks are never re-downloaded. Tests mock the HTTP
layer — no test may hit the live API.

Run as a module to download the configured site (edit the CONFIG block)::

    uv run python -m hybrid_vpp.data.renewables_ninja
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from hybrid_vpp.config.models import PVConfig, WindConfig

log = logging.getLogger(__name__)

API_BASE = "https://www.renewables.ninja/api/data"
TOKEN_ENV_VAR = "RENEWABLES_NINJA_TOKEN"
#: free registered accounts: burst 6/min — stay well below
SECONDS_BETWEEN_REQUESTS = 12.0


def _token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        # fall back to gitignored env files in the project root
        for env_file in (Path("local.env"), Path(".env")):
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith(f"{TOKEN_ENV_VAR}="):
                        token = line.split("=", 1)[1].strip()
                        break
            if token:
                break
    if not token:
        raise RuntimeError(
            f"set {TOKEN_ENV_VAR} (env var or local.env) to your renewables.ninja "
            "API token (https://www.renewables.ninja/profile). Never commit it."
        )
    return token


def _wind_params(cfg: WindConfig) -> dict:
    return {
        "lat": cfg.latitude,
        "lon": cfg.longitude,
        "capacity": cfg.capacity_mw * 1000.0,  # API expects kW
        "height": cfg.hub_height_m,
        "turbine": cfg.turbine_model,
        "format": "json",
        "raw": "false",
    }


def _pv_params(cfg: PVConfig) -> dict:
    return {
        "lat": cfg.latitude,
        "lon": cfg.longitude,
        "capacity": cfg.capacity_mw * 1000.0,  # kW
        "system_loss": cfg.system_loss,
        "tilt": cfg.tilt_deg,
        "azim": cfg.azimuth_deg,
        "tracking": cfg.tracking,
        "format": "json",
        "raw": "false",
    }


def download_year(
    technology: str,
    params: dict,
    year: int,
    out_dir: Path,
    session: requests.Session | None = None,
) -> Path:
    """Download one calendar year for one technology; returns the raw JSON path.

    Skips the request entirely if the raw file is already cached.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{technology}_{year}_raw.json"
    if raw_path.exists():
        log.info("cache hit: %s", raw_path.name)
        return raw_path

    sess = session or requests.Session()
    sess.headers.setdefault("Authorization", f"Token {_token()}")
    query = {**params, "date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
    url = f"{API_BASE}/{technology}"
    log.info("requesting %s %s", url, {k: v for k, v in query.items() if k != "capacity"})
    response = sess.get(url, params=query, timeout=120)
    response.raise_for_status()
    raw_path.write_text(response.text)

    meta = {
        "url": url,
        "params": query,
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "status_code": response.status_code,
    }
    (out_dir / f"{technology}_{year}_meta.json").write_text(json.dumps(meta, indent=2))
    time.sleep(SECONDS_BETWEEN_REQUESTS)
    return raw_path


def parse_raw(raw_path: Path) -> pd.Series:
    """Parse a raw API response into an hourly UTC MW series.

    The JSON ``data`` mapping is keyed by epoch **milliseconds**.
    """
    payload = json.loads(raw_path.read_text())
    frame = pd.DataFrame(payload["data"]).T
    idx = pd.to_datetime(frame.index.astype("int64"), unit="ms", utc=True)
    series = pd.Series(frame["electricity"].astype(float).to_numpy(), index=idx)
    return series.sort_index() / 1000.0  # kW -> MW


def to_quarter_hourly(hourly_mw: pd.Series) -> pd.Series:
    """Upsample hourly mean power to the canonical 15-min grid.

    Linear interpolation between hourly points (documented smoothing
    approximation; energy deviation vs. the hourly means is negligible for
    profile purposes).
    """
    grid = pd.date_range(
        hourly_mw.index.min(), hourly_mw.index.max() + pd.Timedelta(minutes=45), freq="15min"
    )
    return hourly_mw.reindex(grid).interpolate(method="time", limit_direction="both")


def download_site(
    wind: WindConfig,
    pv: PVConfig,
    years: range,
    out_dir: Path,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download (or load cached) site profiles and build the combined parquet.

    Returns a 15-min UTC DataFrame with ``wind_avail_mw`` / ``pv_avail_mw``,
    validated for gaps.
    """
    out_dir = Path(out_dir)
    combined_path = out_dir / "site_profiles_15min.parquet"

    parts: dict[str, pd.Series] = {}
    for technology, params in (("wind", _wind_params(wind)), ("pv", _pv_params(pv))):
        chunks = [
            parse_raw(download_year(technology, params, year, out_dir, session)) for year in years
        ]
        hourly = pd.concat(chunks).sort_index()
        hourly = hourly[~hourly.index.duplicated(keep="first")]
        parts[f"{technology}_avail_mw"] = to_quarter_hourly(hourly)

    df = pd.DataFrame(parts)
    gaps = df.isna().sum()
    if gaps.any():
        log.warning("site profiles contain gaps: %s", gaps.to_dict())
    expected = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    missing = expected.difference(df.index)
    if len(missing):
        log.warning("site profiles missing %d grid intervals", len(missing))
    df.to_parquet(combined_path)
    log.info("wrote %s (%d rows)", combined_path, len(df))
    return df


# --------------------------------------------------------------------------
# CONFIG — edit these constants, then run:
#   uv run python -m hybrid_vpp.data.renewables_ninja
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
YEARS = range(2023, 2027)
OUTPUT_DIR = Path("data/renewables")

if __name__ == "__main__":
    from hybrid_vpp.config.models import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(CONFIG_PATH)
    download_site(cfg.site.wind, cfg.site.pv, YEARS, OUTPUT_DIR)
