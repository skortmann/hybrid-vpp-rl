"""Battery energy storage system model.

Sign convention: **positive BESS power = discharge** (adds to grid export),
negative = charge. A single signed power variable makes simultaneous
charging and discharging impossible by construction.

SoC transition for applied power ``p`` over interval of ``h`` hours::

    E' = E * (1 - self_discharge_per_hour * h)         # optional self-discharge
    E' = E' + h * (eta_ch * max(-p, 0) - max(p, 0) / eta_dis)

Feasible signed-power bounds for an interval are derived from both the power
ratings and the SoC window, so an in-bounds request can never violate the
energy limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from hybrid_vpp.config.models import BatteryConfig
from hybrid_vpp.core.timegrid import INTERVAL, duration_hours


@dataclass(frozen=True, slots=True)
class BatteryInterval:
    """Outcome of applying one interval of battery operation."""

    requested_power_mw: float
    applied_power_mw: float
    charge_energy_mwh: float  # energy drawn at the terminals (>= 0)
    discharge_energy_mwh: float  # energy delivered at the terminals (>= 0)
    throughput_mwh: float
    energy_before_mwh: float
    energy_after_mwh: float
    clipped_mw: float
    clip_reason: str | None


class Battery:
    def __init__(self, config: BatteryConfig) -> None:
        self.cfg = config
        self.capacity_mwh = config.energy_capacity_mwh
        self.energy_min_mwh = config.soc_min * self.capacity_mwh
        self.energy_max_mwh = config.soc_max * self.capacity_mwh
        self.energy_mwh = config.soc_initial * self.capacity_mwh
        self.total_throughput_mwh = 0.0

    @property
    def soc(self) -> float:
        return self.energy_mwh / self.capacity_mwh

    def reset(self) -> None:
        self.energy_mwh = self.cfg.soc_initial * self.capacity_mwh
        self.total_throughput_mwh = 0.0

    def _decayed_energy(self, duration: timedelta) -> float:
        h = duration_hours(duration)
        return self.energy_mwh * (1.0 - self.cfg.self_discharge_per_hour * h)

    def power_bounds(self, duration: timedelta = INTERVAL) -> tuple[float, float]:
        """Feasible signed power ``(p_min, p_max)`` for one interval.

        ``p_max`` (discharge) is limited by the rating and the energy above
        ``soc_min``; ``p_min`` (negative = charge) by the rating and the
        headroom below ``soc_max``. Self-discharge is accounted first.
        """
        h = duration_hours(duration)
        energy = self._decayed_energy(duration)
        p_dis_max = min(
            self.cfg.discharge_power_mw,
            max(0.0, (energy - self.energy_min_mwh)) * self.cfg.discharge_efficiency / h,
        )
        p_ch_max = min(
            self.cfg.charge_power_mw,
            max(0.0, (self.energy_max_mwh - energy)) / (h * self.cfg.charge_efficiency),
        )
        return -p_ch_max, p_dis_max

    def apply(self, power_mw: float, duration: timedelta = INTERVAL) -> BatteryInterval:
        """Apply a signed power request for one interval, clipping to feasibility.

        Clipping is never silent: the returned record carries the requested
        and applied power, the clipped amount, and the reason.
        """
        h = duration_hours(duration)
        p_min, p_max = self.power_bounds(duration)
        applied = min(max(power_mw, p_min), p_max)
        clipped = power_mw - applied
        reason = None
        if abs(clipped) > 1e-12:
            reason = (
                "discharge limited by power rating / SoC floor"
                if power_mw > applied
                else "charge limited by power rating / SoC ceiling"
            )

        energy_before = self.energy_mwh
        energy = self._decayed_energy(duration)
        charge_mw = max(-applied, 0.0)
        discharge_mw = max(applied, 0.0)
        energy += h * (
            self.cfg.charge_efficiency * charge_mw - discharge_mw / self.cfg.discharge_efficiency
        )
        # numerical guard only — bounds above make violations impossible
        self.energy_mwh = min(max(energy, self.energy_min_mwh - 1e-9), self.energy_max_mwh + 1e-9)

        charge_mwh = charge_mw * h
        discharge_mwh = discharge_mw * h
        self.total_throughput_mwh += charge_mwh + discharge_mwh
        return BatteryInterval(
            requested_power_mw=power_mw,
            applied_power_mw=applied,
            charge_energy_mwh=charge_mwh,
            discharge_energy_mwh=discharge_mwh,
            throughput_mwh=charge_mwh + discharge_mwh,
            energy_before_mwh=energy_before,
            energy_after_mwh=self.energy_mwh,
            clipped_mw=clipped,
            clip_reason=reason,
        )
