"""Site-level available wind/PV power profiles.

Two sources, selected by ``DataConfig.site_profile_source``:

* ``renewables_ninja`` — real site profiles previously downloaded by
  :mod:`hybrid_vpp.data.renewables_ninja` (preferred).
* ``zone_scaled`` — **synthetic fallback**: DE-LU zone-level actual
  generation (ENTSO-E) rescaled to the configured site capacities via the
  historical zone maximum. Zone aggregates are much smoother than a single
  site, so variability is understated — clearly a placeholder until real
  profiles are downloaded. It exists so the full pipeline runs without the
  Renewables.ninja token.

Profiles represent *available* (pre-curtailment) power. The physical layer
decides how much of it is exported, stored, or curtailed — profiles are
never pre-clipped to the grid-connection capacity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import DataConfig, SiteConfig
from hybrid_vpp.data.sqlite_market_data import MarketDataStore

log = logging.getLogger(__name__)


def load_site_profiles(
    data_cfg: DataConfig, site: SiteConfig, store: MarketDataStore
) -> pd.DataFrame:
    """15-min UTC DataFrame: ``wind_avail_mw``, ``pv_avail_mw``."""
    if data_cfg.site_profile_source == "renewables_ninja":
        path = Path(data_cfg.renewables_dir) / "site_profiles_15min.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run python -m hybrid_vpp.data.renewables_ninja` "
                "first (requires RENEWABLES_NINJA_TOKEN), or set "
                "data.site_profile_source: zone_scaled"
            )
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        return df

    log.info("using zone-scaled synthetic site profiles (renewables.ninja not configured)")
    return zone_scaled_profiles(site, store)


def zone_scaled_profiles(site: SiteConfig, store: MarketDataStore) -> pd.DataFrame:
    zone = store.zone_renewables()
    wind_tech = "wind_offshore" if site.wind.offshore else "wind_onshore"
    wind_cf = zone[f"actual_{wind_tech}"] / zone[f"actual_{wind_tech}"].max()
    pv_cf = zone["actual_solar"] / zone["actual_solar"].max()
    df = pd.DataFrame(
        {
            "wind_avail_mw": (wind_cf * site.wind.capacity_mw).clip(lower=0.0),
            "pv_avail_mw": (pv_cf * site.pv.capacity_mw).clip(lower=0.0),
        }
    )
    return df.ffill(limit=4).fillna(0.0)
