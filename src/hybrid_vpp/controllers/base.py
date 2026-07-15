"""Controller interface and episode runner for the deterministic simulator."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from hybrid_vpp.markets.calendar import MarketEvent
from hybrid_vpp.sim.simulator import Action, Simulator, StepResult


class Controller(Protocol):
    """Maps a decision event (plus simulator state) to an action.

    Controllers may inspect the simulator's *own* state (positions, SoC,
    site profiles at/ before the event time) and forecast providers.
    They must not read realized future data — providers enforce this for
    forecasts; controllers are audited by the leakage tests.
    """

    def act(self, event: MarketEvent, sim: Simulator) -> Action: ...

    def reset(self) -> None: ...


def run_episode(
    sim: Simulator,
    controller: Controller,
    first_delivery_day: pd.Timestamp | str,
    days: int = 1,
) -> list[StepResult]:
    """Run one full episode under a controller; returns all step results."""
    controller.reset()
    results: list[StepResult] = []
    event = sim.start_episode(pd.Timestamp(first_delivery_day), days=days)
    while event is not None:
        action = controller.act(event, sim)
        result, event = sim.step(action)
        results.append(result)
    return results
