"""Site-level renewable forecast providers.

``ZoneErrorTransferForecast`` is the default: it produces a synthetic site
forecast by transferring the *concurrent DE-LU zone forecast error* (in
capacity-factor terms) onto the site's actual profile::

    site_fc(t) = clip( site_actual(t) + err_cf(t) * site_capacity, 0, capacity )
    err_cf(t)  = (zone_fc(t) - zone_actual(t)) / zone_max

so the agent sees "actual + realistic, weather-correlated error", never the
actual itself. The construction has no fitted parameters (nothing can leak
from validation/test periods into training). Before delivery-day 00:00
local the ENTSO-E *day-ahead* zone forecast is used, afterwards the
*intraday* zone forecast.

Known approximations (documented, config-visible):

* The ENTSO-E day-ahead RES forecast is formally published by 18:00 D-1 —
  a few hours after the DAA gate. Using it at the 12:00 gate is slightly
  optimistic; traders would use an own forecast of similar quality.
* The database stores one intraday-forecast snapshot per interval, not a
  full issue-time history, so the intraday error does not shrink as
  delivery approaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import SiteConfig
from hybrid_vpp.core.timegrid import MARKET_TZ


class PerfectForesightForecast:
    """Returns the realized site profile. Upper-bound benchmarking ONLY."""

    is_perfect_foresight = True

    def __init__(self, site_profiles: pd.DataFrame) -> None:
        self.profiles = site_profiles

    def forecast(self, issue_time, delivery_times) -> pd.DataFrame:
        rows = self.profiles.reindex(delivery_times)
        return pd.DataFrame(
            {"wind_mw": rows["wind_avail_mw"], "pv_mw": rows["pv_avail_mw"]},
            index=delivery_times,
        )


class PersistenceForecast:
    """Seasonal-naive: the site profile exactly ``lag`` earlier (default 24 h).

    Only uses data strictly before the issue time; falls back to further
    lags when the lagged interval itself lies after the issue time.
    """

    def __init__(self, site_profiles: pd.DataFrame, lag: pd.Timedelta | None = None):
        self.profiles = site_profiles
        self.lag = lag if lag is not None else pd.Timedelta("24h")

    def forecast(self, issue_time, delivery_times) -> pd.DataFrame:
        lags = [self.lag * k for k in (1, 2, 3)]
        out = pd.DataFrame(np.nan, index=delivery_times, columns=["wind_mw", "pv_mw"])
        for lag in lags:
            source_times = delivery_times - lag
            usable = source_times < issue_time  # strictly past observations only
            rows = self.profiles.reindex(source_times[usable])
            out.loc[usable, "wind_mw"] = out.loc[usable, "wind_mw"].fillna(
                pd.Series(rows["wind_avail_mw"].to_numpy(), index=delivery_times[usable])
            )
            out.loc[usable, "pv_mw"] = out.loc[usable, "pv_mw"].fillna(
                pd.Series(rows["pv_avail_mw"].to_numpy(), index=delivery_times[usable])
            )
        return out.fillna(0.0)


class NoisyOracleForecast:
    """Actual profile + white noise. Environment debugging ONLY."""

    def __init__(self, site_profiles: pd.DataFrame, site: SiteConfig, sigma: float, seed: int = 0):
        self.profiles = site_profiles
        self.site = site
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def forecast(self, issue_time, delivery_times) -> pd.DataFrame:
        rows = self.profiles.reindex(delivery_times)
        wind_noise = self._rng.normal(0, self.sigma * self.site.wind.capacity_mw, len(rows))
        pv_noise = self._rng.normal(0, self.sigma * self.site.pv.capacity_mw, len(rows))
        return pd.DataFrame(
            {
                "wind_mw": (rows["wind_avail_mw"] + wind_noise).clip(0, self.site.wind.capacity_mw),
                "pv_mw": (rows["pv_avail_mw"] + pv_noise).clip(0, self.site.pv.capacity_mw),
            },
            index=delivery_times,
        )


class ZoneErrorTransferForecast:
    """Synthetic site forecast from concurrent zone forecast errors (default)."""

    def __init__(self, site_profiles: pd.DataFrame, zone: pd.DataFrame, site: SiteConfig) -> None:
        self.profiles = site_profiles
        self.site = site
        wind_tech = "wind_offshore" if site.wind.offshore else "wind_onshore"
        wind_max = zone[f"actual_{wind_tech}"].max()
        solar_max = zone["actual_solar"].max()
        self._err_cf = pd.DataFrame(
            {
                "wind_da": (zone[f"fc_da_{wind_tech}"] - zone[f"actual_{wind_tech}"]) / wind_max,
                "wind_id": (zone[f"fc_id_{wind_tech}"] - zone[f"actual_{wind_tech}"]) / wind_max,
                "pv_da": (zone["fc_da_solar"] - zone["actual_solar"]) / solar_max,
                "pv_id": (zone["fc_id_solar"] - zone["actual_solar"]) / solar_max,
            }
        ).fillna(0.0)

    def forecast(self, issue_time, delivery_times) -> pd.DataFrame:
        rows = self.profiles.reindex(delivery_times)
        err = self._err_cf.reindex(delivery_times).fillna(0.0)
        # intraday zone forecast only once the delivery day has started locally
        issue_local_day = issue_time.tz_convert(MARKET_TZ).normalize()
        delivery_local_day = delivery_times.tz_convert(MARKET_TZ).normalize()
        use_intraday = pd.Series(delivery_local_day <= issue_local_day, index=delivery_times)

        wind_err = err["wind_id"].where(use_intraday, err["wind_da"])
        pv_err = err["pv_id"].where(use_intraday, err["pv_da"])
        wind = rows["wind_avail_mw"] + wind_err * self.site.wind.capacity_mw
        pv = rows["pv_avail_mw"] + pv_err * self.site.pv.capacity_mw
        return pd.DataFrame(
            {
                "wind_mw": wind.clip(0, self.site.wind.capacity_mw),
                "pv_mw": pv.clip(0, self.site.pv.capacity_mw),
            },
            index=delivery_times,
        )


def build_renewable_forecaster(
    mode: str,
    site_profiles: pd.DataFrame,
    zone: pd.DataFrame,
    site: SiteConfig,
    sigma: float = 0.1,
    seed: int = 0,
):
    if mode == "perfect":
        return PerfectForesightForecast(site_profiles)
    if mode == "persistence":
        return PersistenceForecast(site_profiles)
    if mode == "noisy_oracle":
        return NoisyOracleForecast(site_profiles, site, sigma, seed)
    if mode == "zone_scaled_error":
        return ZoneErrorTransferForecast(site_profiles, zone, site)
    raise ValueError(f"unknown renewable forecast mode {mode!r}")
