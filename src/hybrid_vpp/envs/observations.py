"""Observation builder — only information available at the decision time.

Layout (all float32, fixed size):

    scalars:
        event one-hot (6), local-time encodings (4), battery (3),
        grid/site scalars (4), current-interval physicals (6)
    per-slot arrays (n_slots each):
        wind forecast, pv forecast, price reference, net position, action mask

Leakage rules enforced here:

* renewable values come from the forecast provider (issue time = event time);
  realized available power appears only for the *current* dispatch interval;
* the per-slot price reference uses realized prices only where their
  publication time precedes the event (HistoricalPriceView), otherwise the
  price forecast for the event's market;
* normalization uses fixed configuration-derived scales (capacities,
  limits, a global price scale) — nothing is fitted on data, so no
  statistic can leak across the chronological split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.core.timegrid import MARKET_TZ
from hybrid_vpp.envs.actions import ActionLayout
from hybrid_vpp.forecasts.price import HistoricalPriceView
from hybrid_vpp.markets.calendar import EventType, MarketEvent
from hybrid_vpp.sim.simulator import Simulator

PRICE_SCALE = 100.0  # EUR/MWh
PRICE_CLIP = 10.0  # +- 1000 EUR/MWh after scaling

EVENT_ORDER = (
    EventType.DAA_GATE_CLOSURE,
    EventType.IDA1_GATE_CLOSURE,
    EventType.IDA2_GATE_CLOSURE,
    EventType.IDA3_GATE_CLOSURE,
    EventType.IDC_DECISION,
    EventType.PHYSICAL_DISPATCH,
)

N_SCALARS = len(EVENT_ORDER) + 4 + 3 + 4 + 6


@dataclass
class ObservationBuilder:
    cfg: ExperimentConfig
    layout: ActionLayout
    renewable_forecaster: object
    price_forecaster: object
    price_view: HistoricalPriceView
    _slot_times: pd.DatetimeIndex = field(default=None, init=False)

    @property
    def size(self) -> int:
        return N_SCALARS + 5 * self.layout.n_slots

    def start_episode(self, window_start_utc: pd.Timestamp) -> None:
        self._window_start = window_start_utc
        step = pd.Timedelta(hours=1) if self.layout.hourly else pd.Timedelta(minutes=15)
        self._slot_times = pd.DatetimeIndex(
            [window_start_utc + k * step for k in range(self.layout.n_slots)]
        )

    def build(self, event: MarketEvent, sim: Simulator) -> np.ndarray:
        site = self.cfg.site
        cap = site.installed_generation_mw
        t = event.time_utc

        # ---- scalars
        one_hot = np.array([float(event.type == e) for e in EVENT_ORDER])
        local = t.tz_convert(MARKET_TZ)
        tod = local.hour * 60 + local.minute
        time_feats = np.array(
            [
                np.sin(2 * np.pi * tod / 1440.0),
                np.cos(2 * np.pi * tod / 1440.0),
                np.sin(2 * np.pi * local.dayofweek / 7.0),
                np.cos(2 * np.pi * local.dayofweek / 7.0),
            ]
        )
        p_min, p_max = sim.battery.power_bounds()
        battery = np.array(
            [
                sim.battery.soc,
                p_min / site.battery.charge_power_mw,
                p_max / site.battery.discharge_power_mw,
            ]
        )
        grid_scalars = np.array(
            [
                site.grid.export_limit_mw / cap,
                site.grid.import_limit_mw / cap,
                site.oversizing_ratio - 1.0,
                (sim.battery.energy_max_mwh - sim.battery.energy_mwh)
                / site.battery.energy_capacity_mwh,  # charge headroom (energy)
            ]
        )

        current = np.zeros(6)
        if event.type == EventType.PHYSICAL_DISPATCH:
            product = event.products[0]
            row = sim.profiles.loc[product.start_utc]
            wind, pv = float(row["wind_avail_mw"]), float(row["pv_avail_mw"])
            position = sim.book.net_position_mw(product)
            excess = max(0.0, wind + pv - site.grid.export_limit_mw)
            headroom_mw = -p_min
            current = np.array(
                [
                    wind / site.wind.capacity_mw,
                    pv / site.pv.capacity_mw,
                    position / site.grid.export_limit_mw,
                    excess / cap,
                    headroom_mw / site.battery.charge_power_mw,
                    max(0.0, excess - headroom_mw) / cap,  # expected forced curtailment
                ]
            )

        # ---- per-slot arrays
        fc = self.renewable_forecaster.forecast(t, self._slot_times)
        wind_fc = (fc["wind_mw"].to_numpy(float) / site.wind.capacity_mw).clip(0, 1)
        pv_fc = (fc["pv_mw"].to_numpy(float) / site.pv.capacity_mw).clip(0, 1)

        market = event.market or "idc"
        price_fc = self.price_forecaster.forecast(market, t, self._slot_times)
        published_daa = self.price_view.visible("daa", t).reindex(self._slot_times)
        price_ref = published_daa.fillna(price_fc)
        prices = (price_ref.to_numpy(float) / PRICE_SCALE).clip(-PRICE_CLIP, PRICE_CLIP)

        positions = (
            np.array([sim.book.net_position_mw_at(ts) for ts in self._slot_times])
            / site.grid.export_limit_mw
        )

        mask = self.layout.mask(self._window_start, event)[: self.layout.n_slots]

        obs = np.concatenate(
            [
                one_hot,
                time_feats,
                battery,
                grid_scalars,
                current,
                wind_fc,
                pv_fc,
                prices,
                positions,
                mask,
            ]
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
