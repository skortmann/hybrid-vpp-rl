"""Gymnasium environment wrapping the deterministic VPP simulator.

One episode = ``episode.days`` consecutive local delivery days, sampled
from the configured chronological split. The agent acts at every decision
event (auction gates, IDC decisions, physical dispatch) through a
fixed-size masked action vector (see :mod:`hybrid_vpp.envs.actions`).

Reward = cash flows booked during the step (EUR, scaled by 1e-3), i.e.
market cash at execution and settlement components at delivery — each euro
rewarded exactly once — plus an optional penalty on infeasible requested
actions, plus a terminal valuation of the battery's residual energy at the
episode's mean day-ahead price (so holding energy at episode end is not
punished; both are documented reward-design choices).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.core.timegrid import energy_mwh, local_day_bounds_utc
from hybrid_vpp.data.site_profiles import load_site_profiles
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.envs.actions import ActionLayout
from hybrid_vpp.envs.observations import ObservationBuilder
from hybrid_vpp.forecasts.price import HistoricalPriceView, build_price_forecaster
from hybrid_vpp.forecasts.renewable import build_renewable_forecaster
from hybrid_vpp.markets.calendar import MarketEvent
from hybrid_vpp.sim.simulator import Simulator

REWARD_SCALE = 1e-3  # EUR -> kEUR


class HybridVppEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ExperimentConfig,
        split: str = "train",
        store: MarketDataStore | None = None,
        profiles: pd.DataFrame | None = None,
        sequential_days: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.sequential_days = sequential_days
        store = store or MarketDataStore(cfg.data, cfg.markets, cfg.synthetic_market)
        profiles = (
            profiles if profiles is not None else load_site_profiles(cfg.data, cfg.site, store)
        )
        self.sim = Simulator(cfg, store, profiles)

        price_series = {
            "daa": store.daa_prices()["price_eur_per_mwh"],
            "ida1": store.ida_prices("ida1"),
            "ida2": store.ida_prices("ida2"),
            "ida3": store.ida_prices("ida3"),
            "idc": store.idc_indices()["IDFULL"],
        }
        renewable_fc = build_renewable_forecaster(
            cfg.forecast.renewable_mode,
            profiles,
            store.zone_renewables(),
            cfg.site,
            sigma=cfg.forecast.noisy_oracle_sigma,
        )
        price_fc = build_price_forecaster(cfg.forecast.price_mode, self.sim.calendar, price_series)
        self.layout = ActionLayout(cfg, cfg.episode.days)
        self.obs_builder = ObservationBuilder(
            cfg,
            self.layout,
            renewable_fc,
            price_fc,
            HistoricalPriceView(self.sim.calendar, price_series),
        )
        self._daa_prices = price_series["daa"]

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.layout.size,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_builder.size,), np.float32
        )
        self.valid_days = self._valid_days(store, profiles)
        if not len(self.valid_days):
            raise ValueError(f"no valid episode days in split {split!r}")
        self._day_cursor = 0
        self._event: MarketEvent | None = None

    # ------------------------------------------------------------------ days

    def _valid_days(self, store: MarketDataStore, profiles: pd.DataFrame) -> pd.DatetimeIndex:
        """Delivery days in the split with complete DAA, reBAP, and profiles."""
        s = self.cfg.split
        start = {"train": s.train_start, "val": s.val_start, "test": s.test_start}[self.split]
        end = {"train": s.train_end, "val": s.val_end, "test": s.test_end}[self.split]
        days = pd.date_range(start, end - pd.Timedelta(days=self.cfg.episode.days), freq="D")
        daa = store.daa_prices()
        rebap = store.rebap()
        keep = []
        for day in days:
            try:
                w0, w1 = local_day_bounds_utc(day)
                w1 = local_day_bounds_utc(day + pd.Timedelta(days=self.cfg.episode.days - 1))[1]
            except Exception:
                continue
            n_qh = int((w1 - w0) / pd.Timedelta(minutes=15))
            grid = pd.date_range(w0, w1, freq="15min", inclusive="left")
            if (
                len(daa.loc[w0 : w1 - pd.Timedelta(minutes=1)]) > 0
                and rebap.reindex(grid).notna().all()
                and profiles.reindex(grid).notna().all().all()
                and len(profiles.reindex(grid)) == n_qh
            ):
                keep.append(day)
        return pd.DatetimeIndex(keep)

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "day" in options:
            day = pd.Timestamp(options["day"])
        elif self.sequential_days:
            day = self.valid_days[self._day_cursor % len(self.valid_days)]
            self._day_cursor += 1
        else:
            day = self.valid_days[self.np_random.integers(len(self.valid_days))]

        self._event = self.sim.start_episode(day, days=self.cfg.episode.days)
        window_start = local_day_bounds_utc(day)[0]
        self.obs_builder.start_episode(window_start)
        self._window_start = window_start
        self._episode_day = day
        obs = self.obs_builder.build(self._event, self.sim)
        return obs, self._info(self._event)

    def step(self, action: np.ndarray):
        if self._event is None:
            raise RuntimeError("call reset() first")
        event = self._event
        logical = self.layout.translate(action, self._window_start, event, self.sim)
        result, next_event = self.sim.step(logical)

        reward = result.cash_eur
        if (
            self.cfg.episode.infeasibility_penalty_eur_per_mwh
            and result.dispatch_record is not None
        ):
            d = result.dispatch_record.dispatch
            infeasible_mw = (
                abs(d.requested_bess_power_mw - d.bess_power_mw)
                + abs(d.requested_wind_curtail_mw - d.wind_curtail_mw)
                + abs(d.requested_pv_curtail_mw - d.pv_curtail_mw)
            )
            reward -= self.cfg.episode.infeasibility_penalty_eur_per_mwh * energy_mwh(
                infeasible_mw, result.dispatch_record.product.duration
            )

        terminated = next_event is None
        if terminated:
            reward += self._terminal_soc_value()
        self._event = next_event

        obs = (
            self.obs_builder.build(next_event, self.sim)
            if next_event is not None
            else np.zeros(self.obs_builder.size, dtype=np.float32)
        )
        info = self._info(next_event) if next_event is not None else self._final_info()
        return obs, reward * REWARD_SCALE, terminated, False, info

    def _terminal_soc_value(self) -> float:
        """Value residual battery energy vs. episode start at the mean DA price."""
        w0, _ = local_day_bounds_utc(self._episode_day)
        _, w1 = local_day_bounds_utc(
            self._episode_day + pd.Timedelta(days=self.cfg.episode.days - 1)
        )
        mean_price = float(self._daa_prices.loc[w0:w1].mean())
        e0 = self.cfg.site.battery.soc_initial * self.cfg.site.battery.energy_capacity_mwh
        return (self.sim.battery.energy_mwh - e0) * mean_price

    def _info(self, event: MarketEvent) -> dict[str, Any]:
        return {
            "event_type": event.type.name,
            "event_time": event.time_utc,
            "action_mask": self.layout.mask(self._window_start, event),
        }

    def _final_info(self) -> dict[str, Any]:
        from hybrid_vpp.evaluation.metrics import episode_metrics

        return {"episode_metrics": episode_metrics(self.sim), "action_mask": None}
