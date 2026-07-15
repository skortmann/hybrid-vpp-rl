"""Typed configuration models for the hybrid VPP framework.

Every tunable of the simulator, markets, forecasts, and training is declared
here and validated at startup. Configurations are loaded from YAML via
:func:`load_config`. No magic numbers may live in the environment code —
they belong here, with documentation.
"""

from __future__ import annotations

import os
import re
from datetime import time, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- site


class WindConfig(BaseModel):
    """Wind park parameters (site + Renewables.ninja request parameters)."""

    capacity_mw: float = Field(gt=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    hub_height_m: float = Field(default=100.0, gt=0)
    turbine_model: str = "Vestas V112 3000"
    offshore: bool = False


class PVConfig(BaseModel):
    """PV park parameters (site + Renewables.ninja request parameters)."""

    capacity_mw: float = Field(gt=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    system_loss: float = Field(default=0.1, ge=0, le=1)
    tilt_deg: float = Field(default=35.0, ge=0, le=90)
    azimuth_deg: float = Field(default=180.0, ge=0, le=360)
    tracking: Literal[0, 1, 2] = 0  # 0=fixed, 1=single-axis, 2=dual-axis


class CongestionWeights(BaseModel):
    """Quadratic deviation weights of the feasibility projection (mode=optimization)."""

    bess_deviation: float = Field(default=1.0, gt=0)
    wind_curtailment: float = Field(default=10.0, gt=0)
    pv_curtailment: float = Field(default=10.0, gt=0)


CongestionMode = Literal[
    "optimization",
    "battery_first",
    "curtailment_first",
    "pv_first",
    "wind_first",
    "proportional_curtailment",
]


class CongestionResolutionConfig(BaseModel):
    """How requested dispatch is projected onto the feasible set at the PCC."""

    mode: CongestionMode = "optimization"
    weights: CongestionWeights = CongestionWeights()


class GridConnectionConfig(BaseModel):
    """Common point-of-connection limits. Export is positive grid power."""

    export_limit_mw: float = Field(gt=0)
    import_limit_mw: float = Field(default=0.0, ge=0)
    allow_grid_import: bool = True
    congestion_resolution: CongestionResolutionConfig = CongestionResolutionConfig()

    @model_validator(mode="after")
    def _import_consistent(self) -> GridConnectionConfig:
        if not self.allow_grid_import and self.import_limit_mw > 0:
            raise ValueError("import_limit_mw > 0 requires allow_grid_import=true")
        return self


class BatteryConfig(BaseModel):
    """BESS ratings and operating window.

    Sign convention: positive BESS power = discharge (adds to grid export).
    """

    energy_capacity_mwh: float = Field(gt=0)
    charge_power_mw: float = Field(gt=0)
    discharge_power_mw: float = Field(gt=0)
    charge_efficiency: float = Field(default=0.95, gt=0, le=1)
    discharge_efficiency: float = Field(default=0.95, gt=0, le=1)
    soc_min: float = Field(default=0.05, ge=0, lt=1)
    soc_max: float = Field(default=0.95, gt=0, le=1)
    soc_initial: float = Field(default=0.5, ge=0, le=1)
    soc_terminal_target: float | None = Field(default=None, ge=0, le=1)
    self_discharge_per_hour: float = Field(default=0.0, ge=0, lt=0.1)
    cycle_throughput_limit_mwh_per_day: float | None = Field(default=None, gt=0)
    degradation_cost_eur_per_mwh_throughput: float = Field(default=1.5, ge=0)

    @model_validator(mode="after")
    def _soc_window(self) -> BatteryConfig:
        if self.soc_min >= self.soc_max:
            raise ValueError("soc_min must be < soc_max")
        if not (self.soc_min <= self.soc_initial <= self.soc_max):
            raise ValueError("soc_initial must lie within [soc_min, soc_max]")
        if self.soc_terminal_target is not None and not (
            self.soc_min <= self.soc_terminal_target <= self.soc_max
        ):
            raise ValueError("soc_terminal_target must lie within [soc_min, soc_max]")
        return self


class SiteConfig(BaseModel):
    """Physical portfolio: oversized generation behind a constrained PCC."""

    name: str = "hybrid-park"
    wind: WindConfig
    pv: PVConfig
    battery: BatteryConfig
    grid: GridConnectionConfig
    curtailment_cost_eur_per_mwh: float = Field(default=0.0, ge=0)

    @property
    def installed_generation_mw(self) -> float:
        return self.wind.capacity_mw + self.pv.capacity_mw

    @property
    def oversizing_ratio(self) -> float:
        return self.installed_generation_mw / self.grid.export_limit_mw

    @property
    def excess_capacity_mw(self) -> float:
        return self.installed_generation_mw - self.grid.export_limit_mw


# ------------------------------------------------------------------------ markets


class AuctionSessionConfig(BaseModel):
    """One auction session (DAA or an IDA), times in Europe/Berlin wall clock."""

    enabled: bool = True
    gate_closure_local: time
    #: results become visible this many minutes after gate closure
    publication_delay_min: int = Field(default=30, ge=0)
    #: day offset of gate closure relative to delivery day D (-1 = day before)
    gate_day_offset: int = Field(default=-1, le=0)
    #: delivery window covered, as local hours of day D (end exclusive)
    delivery_hours: tuple[int, int] = (0, 24)
    #: per-product volume limit for one submission, MW
    max_volume_mw: float = Field(default=1000.0, gt=0)
    transaction_cost_eur_per_mwh: float = Field(default=0.0, ge=0)


class IdcConfig(BaseModel):
    """Intraday-continuous decision & execution model (price-taker, documented)."""

    enabled: bool = True
    #: cadence of IDC decision events on the canonical grid
    decision_frequency: timedelta = timedelta(hours=1)
    #: products of delivery day D open at this local wall time on day D-1
    opening_local_time: time = time(16, 0)
    #: trading closes `gate_closure_lead` before delivery start
    #: (30 min ~ cross-zonal gate; 5 min would model DE local closure)
    gate_closure_lead: timedelta = timedelta(minutes=30)
    #: which historical index executes a trade, by remaining lead time
    execution_index_by_lead: dict[str, str] = {"1h": "ID1", "3h": "ID3", "inf": "IDFULL"}
    max_volume_mw_per_trade: float = Field(default=1000.0, gt=0)
    transaction_cost_eur_per_mwh: float = Field(default=0.1, ge=0)

    @field_validator("decision_frequency", "gate_closure_lead", mode="before")
    @classmethod
    def _parse_td(cls, v: object) -> object:
        return pd.Timedelta(v).to_pytimedelta() if isinstance(v, str) else v


class ImbalanceConfig(BaseModel):
    """Deviation settlement model.

    ``rebap``: settle at the historical 15-min reBAP (single price) — this is
    the actual German imbalance settlement, not an approximation.
    ``symmetric_penalty``/``asymmetric_spread``: stylized alternatives.
    """

    model: Literal["rebap", "symmetric_penalty", "asymmetric_spread"] = "rebap"
    penalty_eur_per_mwh: float = Field(default=100.0, ge=0)
    spread_eur_per_mwh: float = Field(default=25.0, ge=0)
    #: extra additive penalty on |deviation| (0 = pure price settlement)
    deviation_penalty_eur_per_mwh: float = Field(default=0.0, ge=0)


class MarketsConfig(BaseModel):
    bidding_zone: str = "DE_LU"
    #: date from which day-ahead products are 15-minute (SDAC MTU switch)
    daa_quarter_hourly_from: pd.Timestamp = pd.Timestamp("2025-10-01")
    daa: AuctionSessionConfig = AuctionSessionConfig(
        gate_closure_local=time(12, 0), publication_delay_min=60, gate_day_offset=-1
    )
    ida1: AuctionSessionConfig = AuctionSessionConfig(
        gate_closure_local=time(15, 0), gate_day_offset=-1
    )
    ida2: AuctionSessionConfig = AuctionSessionConfig(
        gate_closure_local=time(22, 0), gate_day_offset=-1
    )
    ida3: AuctionSessionConfig = AuctionSessionConfig(
        gate_closure_local=time(10, 0), gate_day_offset=0, delivery_hours=(12, 24)
    )
    idc: IdcConfig = IdcConfig()
    imbalance: ImbalanceConfig = ImbalanceConfig()

    @field_validator("daa_quarter_hourly_from", mode="before")
    @classmethod
    def _parse_ts(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------------- data/splits

MARKET_DB_ENV_VAR = "MARKET_DATABASE_PATH"


class FallbackTriggers(BaseModel):
    """Which real-database problems trigger the synthetic fallback (mode=auto)."""

    missing_file: bool = True
    invalid_schema: bool = True
    insufficient_coverage: bool = True
    #: missing optional series never replace the whole database; they may be
    #: approximated per-series through the composed-data interface
    missing_optional_series: bool = False


class MarketDatabaseConfig(BaseModel):
    """Market-database source selection (real / synthetic / auto)."""

    mode: Literal["real", "synthetic", "auto"] = "auto"
    #: real database path; None -> $MARKET_DATABASE_PATH, then the known default
    path: Path | None = None
    synthetic_path: Path = Path("data/generated/synthetic-iaew-marktdaten.db")
    create_if_missing: bool = True
    fallback_on: FallbackTriggers = FallbackTriggers()

    def resolved_real_path(self) -> Path | None:
        """Configured path, else ``$MARKET_DATABASE_PATH``, else None (no real DB)."""
        if self.path is not None:
            return self.path
        env = os.environ.get(MARKET_DB_ENV_VAR, "").strip()
        return Path(env) if env else None


class SyntheticPricesConfig(BaseModel):
    allow_negative_prices: bool = True
    positive_spikes: bool = True
    negative_spikes: bool = True
    minimum_price_eur_per_mwh: float = -500.0
    maximum_price_eur_per_mwh: float = 1500.0
    #: expected number of spike events per year (per sign)
    spikes_per_year: float = Field(default=40.0, ge=0)
    #: base level and slope of the merit-order price curve
    base_price_eur_per_mwh: float = 70.0
    residual_slope_eur_per_mwh: float = 90.0


class SyntheticIdcConfig(BaseModel):
    generate_bid_ask: bool = True
    minimum_spread_eur_per_mwh: float = Field(default=0.1, ge=0)


class SyntheticImbalanceConfig(BaseModel):
    enabled: bool = True
    #: write different unterdeckt/ueberdeckt prices (single price when False)
    asymmetric: bool = False
    spread_eur_per_mwh: float = Field(default=10.0, ge=0)


class SyntheticCalibrationConfig(BaseModel):
    enabled: bool = False
    #: real database used to derive aggregate calibration parameters
    source_database: Path | None = None


class SyntheticMarketConfig(BaseModel):
    """Deterministic synthetic market-data generator settings."""

    start: pd.Timestamp = pd.Timestamp("2025-01-01")
    end: pd.Timestamp = pd.Timestamp("2026-01-01")  # exclusive
    resolution_minutes: Literal[15] = 15
    timezone: str = "Europe/Berlin"
    bidding_zone: str = "DE_LU"
    random_seed: int = 42
    prices: SyntheticPricesConfig = SyntheticPricesConfig()
    idc: SyntheticIdcConfig = SyntheticIdcConfig()
    imbalance: SyntheticImbalanceConfig = SyntheticImbalanceConfig()
    calibration: SyntheticCalibrationConfig = SyntheticCalibrationConfig()
    #: zone-level scale of the synthetic fundamentals (MW)
    zone_load_mw: float = 55_000.0
    zone_wind_capacity_mw: float = 70_000.0
    zone_pv_capacity_mw: float = 90_000.0

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_ts(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)

    @model_validator(mode="after")
    def _range(self) -> SyntheticMarketConfig:
        if self.start >= self.end:
            raise ValueError("synthetic_market.start must be before end")
        return self

    model_config = {"arbitrary_types_allowed": True}


class DataConfig(BaseModel):
    #: legacy single-path setting; feeds market_database.path when set
    market_db_path: Path | None = None
    market_database: MarketDatabaseConfig = MarketDatabaseConfig()
    cache_dir: Path = Path("data/cache")
    renewables_dir: Path = Path("data/renewables")
    #: site renewable profiles: "renewables_ninja" (requires cached download)
    #: or "zone_scaled" (synthetic fallback from ENTSO-E DE-LU zone actuals)
    site_profile_source: Literal["renewables_ninja", "zone_scaled"] = "zone_scaled"

    @model_validator(mode="after")
    def _legacy_path(self) -> DataConfig:
        if self.market_db_path is not None and self.market_database.path is None:
            self.market_database.path = self.market_db_path
        return self


class SplitConfig(BaseModel):
    """Chronological train/validation/test periods (UTC dates, end exclusive)."""

    train_start: pd.Timestamp = pd.Timestamp("2024-06-14")
    train_end: pd.Timestamp = pd.Timestamp("2025-11-01")
    val_start: pd.Timestamp = pd.Timestamp("2025-11-01")
    val_end: pd.Timestamp = pd.Timestamp("2026-02-01")
    test_start: pd.Timestamp = pd.Timestamp("2026-02-01")
    test_end: pd.Timestamp = pd.Timestamp("2026-05-10")

    @field_validator("*", mode="before")
    @classmethod
    def _parse_ts(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)

    @model_validator(mode="after")
    def _chronological(self) -> SplitConfig:
        order = [
            self.train_start,
            self.train_end,
            self.val_start,
            self.val_end,
            self.test_start,
            self.test_end,
        ]
        if any(a > b for a, b in zip(order, order[1:], strict=False)):
            raise ValueError(f"splits must be chronological and non-overlapping: {order}")
        return self

    def split_of(self, ts: pd.Timestamp) -> Literal["train", "val", "test", "none"]:
        t = pd.Timestamp(ts).tz_localize(None)
        if self.train_start <= t < self.train_end:
            return "train"
        if self.val_start <= t < self.val_end:
            return "val"
        if self.test_start <= t < self.test_end:
            return "test"
        return "none"

    model_config = {"arbitrary_types_allowed": True}


# ------------------------------------------------------------------- forecasts/rl


class ForecastConfig(BaseModel):
    renewable_mode: Literal["persistence", "zone_scaled_error", "noisy_oracle", "perfect"] = (
        "zone_scaled_error"
    )
    price_mode: Literal["seasonal_naive", "rolling_mean", "perfect"] = "seasonal_naive"
    #: standard deviation of noisy-oracle noise as a fraction of capacity (debug only)
    noisy_oracle_sigma: float = Field(default=0.1, ge=0)
    horizon: timedelta = timedelta(hours=48)

    @field_validator("horizon", mode="before")
    @classmethod
    def _parse_td(cls, v: object) -> object:
        return pd.Timedelta(v).to_pytimedelta() if isinstance(v, str) else v


class EpisodeConfig(BaseModel):
    """One episode = `days` consecutive local delivery days."""

    days: int = Field(default=1, ge=1)
    #: penalty applied by the env for requested-but-infeasible actions, EUR/MWh
    infeasibility_penalty_eur_per_mwh: float = Field(default=0.0, ge=0)


class TrainingConfig(BaseModel):
    algorithm: Literal["ppo", "sac"] = "ppo"
    total_timesteps: int = Field(default=2_000_000, gt=0)
    seed: int = 0
    n_envs: int = Field(default=8, ge=1)
    eval_freq: int = Field(default=50_000, gt=0)
    n_eval_episodes: int = Field(default=20, ge=1)
    run_name: str = "ppo-baseline"
    checkpoint_dir: Path = Path("runs")
    tensorboard_dir: Path = Path("runs/tb")
    tracker: Literal["wandb", "tensorboard", "none"] = "wandb"
    wandb_project: str = "hybrid-vpp-rl"
    policy_kwargs: dict = {}
    algo_kwargs: dict = {}


class ExperimentConfig(BaseModel):
    """Root configuration object."""

    site: SiteConfig
    markets: MarketsConfig = MarketsConfig()
    data: DataConfig = DataConfig()
    synthetic_market: SyntheticMarketConfig = SyntheticMarketConfig()
    split: SplitConfig = SplitConfig()
    forecast: ForecastConfig = ForecastConfig()
    episode: EpisodeConfig = EpisodeConfig()
    training: TrainingConfig = TrainingConfig()


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate_env(node: object) -> object:
    """Replace ``${VAR}`` placeholders with environment values (missing -> None)."""
    if isinstance(node, dict):
        return {k: _interpolate_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_env(v) for v in node]
    if isinstance(node, str):
        match = _ENV_PATTERN.fullmatch(node.strip())
        if match:
            return os.environ.get(match.group(1)) or None
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), node)
    return node


def _load_env_files() -> None:
    """Load ``local.env`` / ``.env`` from the working directory (unset vars only).

    Keeps machine-specific settings such as ``MARKET_DATABASE_PATH`` and API
    tokens out of source and configuration files.
    """
    for env_file in (Path("local.env"), Path(".env")):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    ``${ENV_VAR}`` placeholders are substituted from the environment
    (``local.env`` / ``.env`` are loaded first); an unset placeholder
    resolves to null.
    """
    _load_env_files()
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(_interpolate_env(raw))
