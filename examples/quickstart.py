"""Quick start: simulate one delivery day on synthetic market data.

Runs entirely offline with the minimal installation: the synthetic market
database is generated on first use (deterministic per seed), the
rule-based controller trades the day through DAA -> IDA1/2/3 -> IDC, and
the settled economics are printed.

Run as-is:  uv run python examples/quickstart.py
Tunables are the constants below.
"""

from pathlib import Path

# --------------------------------------------------------------------- CONFIG

CONFIG_PATH = Path("configs/synthetic_market.yaml")
DAY = "2025-05-14"

# ------------------------------------------------------------------------


def main(config_path: Path = CONFIG_PATH, day: str = DAY) -> dict:
    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.controllers import RuleBasedController
    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.evaluation.metrics import episode_metrics
    from hybrid_vpp.evaluation.run_baselines import build_stack

    cfg = load_config(config_path)
    _store, _profiles, sim, renewable_fc, price_fc = build_stack(cfg)
    controller = RuleBasedController(cfg, renewable_fc, price_fc)
    run_episode(sim, controller, day)
    metrics = episode_metrics(sim)

    print(f"delivery day {day} (synthetic market data)")
    for key in (
        "total_net_revenue_eur",
        "market_revenue_eur",
        "imbalance_cash_eur",
        "abs_deviation_mwh",
        "grid_export_mwh",
        "equivalent_full_cycles",
        "curtailment_ratio",
    ):
        print(f"  {key:28s} {metrics[key]:12,.2f}")
    return metrics


if __name__ == "__main__":
    main()
