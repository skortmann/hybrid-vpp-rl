"""Battery model: SoC dynamics, efficiency, limits, hand-calculated examples."""

from datetime import timedelta

import pytest

from hybrid_vpp.assets.battery import Battery
from hybrid_vpp.config.models import BatteryConfig

QH = timedelta(minutes=15)
H = timedelta(hours=1)


def make_battery(**kw) -> Battery:
    defaults = dict(
        energy_capacity_mwh=60.0,
        charge_power_mw=30.0,
        discharge_power_mw=30.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
    )
    defaults.update(kw)
    return Battery(BatteryConfig(**defaults))


def test_charge_hand_calculated():
    bat = make_battery()  # E0 = 30 MWh
    r = bat.apply(-20.0, H)  # charge 20 MW for 1 h at eta 0.9 -> +18 MWh
    assert r.applied_power_mw == -20.0
    assert r.charge_energy_mwh == pytest.approx(20.0)
    assert bat.energy_mwh == pytest.approx(48.0)


def test_discharge_hand_calculated():
    bat = make_battery()  # E0 = 30
    r = bat.apply(9.0, H)  # discharge 9 MW for 1 h at eta 0.9 -> -10 MWh from storage
    assert r.discharge_energy_mwh == pytest.approx(9.0)
    assert bat.energy_mwh == pytest.approx(20.0)


def test_quarter_hour_energy_conversion():
    bat = make_battery()
    bat.apply(9.0, QH)  # 9 MW for 15 min = 2.25 MWh delivered, 2.5 MWh from storage
    assert bat.energy_mwh == pytest.approx(30.0 - 2.5)


def test_discharge_clipped_at_soc_floor():
    bat = make_battery(soc_initial=0.12)  # E0 = 7.2, floor 6.0 -> 1.2 MWh above floor
    p_min, p_max = bat.power_bounds(H)
    assert p_max == pytest.approx(1.2 * 0.9)  # limited by energy, not rating
    r = bat.apply(30.0, H)
    assert r.applied_power_mw == pytest.approx(p_max)
    assert r.clip_reason is not None
    assert bat.energy_mwh == pytest.approx(6.0)


def test_charge_clipped_at_soc_ceiling():
    bat = make_battery(soc_initial=0.88)  # E0 = 52.8, ceiling 54 -> 1.2 MWh headroom
    p_min, _ = bat.power_bounds(H)
    assert p_min == pytest.approx(-1.2 / 0.9)
    r = bat.apply(-30.0, H)
    assert r.applied_power_mw == pytest.approx(p_min)
    assert bat.energy_mwh == pytest.approx(54.0)


def test_power_rating_limits():
    bat = make_battery()
    r = bat.apply(100.0, QH)
    assert r.applied_power_mw == 30.0
    r = bat.apply(-100.0, QH)
    assert r.applied_power_mw == -30.0


def test_no_simultaneous_charge_discharge_by_construction():
    bat = make_battery()
    r = bat.apply(-15.0, QH)
    assert r.charge_energy_mwh > 0 and r.discharge_energy_mwh == 0.0
    r = bat.apply(15.0, QH)
    assert r.discharge_energy_mwh > 0 and r.charge_energy_mwh == 0.0


def test_throughput_accumulates():
    bat = make_battery()
    bat.apply(-20.0, H)
    bat.apply(10.0, H)
    assert bat.total_throughput_mwh == pytest.approx(30.0)


def test_self_discharge():
    bat = make_battery(self_discharge_per_hour=0.001)
    bat.apply(0.0, H)
    assert bat.energy_mwh == pytest.approx(30.0 * 0.999)


def test_reset():
    bat = make_battery()
    bat.apply(-20.0, H)
    bat.reset()
    assert bat.energy_mwh == pytest.approx(30.0)
    assert bat.total_throughput_mwh == 0.0
