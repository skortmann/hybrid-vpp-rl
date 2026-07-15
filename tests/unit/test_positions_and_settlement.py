"""Position book, ledger, execution, and settlement accounting (hand-checked)."""

from datetime import timedelta

import pandas as pd
import pytest

from hybrid_vpp.config.models import AuctionSessionConfig, IdcConfig, ImbalanceConfig
from hybrid_vpp.core.timegrid import DeliveryProduct
from hybrid_vpp.markets.execution import (
    execute_auction_orders,
    execute_idc_orders,
    select_idc_index,
)
from hybrid_vpp.markets.positions import Ledger, PositionBook, Trade
from hybrid_vpp.markets.settlement import ImbalanceSettlement

T0 = pd.Timestamp("2025-03-15 12:00", tz="UTC")
QH = DeliveryProduct(T0)
HOURLY = DeliveryProduct(T0, timedelta(hours=1))


def trade(side="sell", volume=10.0, price=50.0, market="daa", executed=None, product=QH):
    return Trade(
        trade_id=0,
        market=market,
        product=product,
        side=side,
        volume_mw=volume,
        price_eur_per_mwh=price,
        executed_utc=executed or T0 - pd.Timedelta(hours=1),
    )


class TestPositionBook:
    def test_signed_positions_and_energy(self):
        book = PositionBook()
        book.add(trade(side="sell", volume=10.0))
        book.add(trade(side="buy", volume=4.0, market="idc"))
        assert book.net_position_mw(QH) == pytest.approx(6.0)
        assert book.contracted_energy_mwh(QH) == pytest.approx(1.5)  # 6 MW * 0.25 h
        assert book.market_breakdown_mw(QH) == pytest.approx(
            {"daa": 10.0, "ida1": 0, "ida2": 0, "ida3": 0, "idc": -4.0}
        )

    def test_retroactive_trade_rejected(self):
        book = PositionBook()
        with pytest.raises(ValueError, match="retroactive"):
            book.add(trade(executed=T0))  # executed at delivery start
        with pytest.raises(ValueError, match="retroactive"):
            book.add(trade(executed=T0 + pd.Timedelta(minutes=5)))

    def test_positions_are_additive_not_overwritten(self):
        book = PositionBook()
        book.add(trade(side="sell", volume=10.0))
        book.add(trade(side="sell", volume=5.0, market="ida1"))
        assert book.net_position_mw(QH) == pytest.approx(15.0)
        assert len(book.trades) == 2  # both fills retained

    def test_cash_flow_signs(self):
        sell = trade(side="sell", volume=10.0, price=50.0)
        buy = trade(side="buy", volume=10.0, price=50.0)
        assert sell.cash_flow_eur == pytest.approx(+125.0)  # 2.5 MWh * 50
        assert buy.cash_flow_eur == pytest.approx(-125.0)


class TestLedger:
    def test_totals_by_component(self):
        led = Ledger()
        led.add(T0, "daa", 100.0)
        led.add(T0, "imbalance", -30.0)
        led.add(T0, "transaction_cost", -1.0)
        assert led.total() == pytest.approx(69.0)
        assert led.by_component()["daa"] == pytest.approx(100.0)

    def test_unknown_component_rejected(self):
        with pytest.raises(ValueError):
            Ledger().add(T0, "windfall", 1.0)


class TestAuctionExecution:
    def session(self, **kw):
        from datetime import time

        defaults = dict(
            gate_closure_local=time(12, 0), max_volume_mw=100.0, transaction_cost_eur_per_mwh=0.1
        )
        defaults.update(kw)
        return AuctionSessionConfig(**defaults)

    def test_hourly_product_fills_four_quarter_hours(self):
        book = PositionBook()
        prices = pd.Series({T0: 80.0})
        reports = execute_auction_orders(
            market="daa",
            session=self.session(),
            event_products=(HOURLY,),
            orders={HOURLY: 20.0},
            clearing_prices=prices,
            gate_utc=T0 - pd.Timedelta(hours=13),
            book=book,
        )
        assert len(reports) == 1 and len(reports[0].trades) == 4
        assert all(t.volume_mw == 20.0 and t.price_eur_per_mwh == 80.0 for t in reports[0].trades)
        # 20 MW for one hour at 80 EUR/MWh = 1600 EUR across the 4 QH fills
        assert sum(t.cash_flow_eur for t in reports[0].trades) == pytest.approx(1600.0)
        assert book.net_position_mw(QH) == pytest.approx(20.0)

    def test_order_outside_scope_raises(self):
        other = DeliveryProduct(T0 + pd.Timedelta(days=1))
        with pytest.raises(ValueError, match="outside auction scope"):
            execute_auction_orders(
                market="daa",
                session=self.session(),
                event_products=(HOURLY,),
                orders={other: 5.0},
                clearing_prices=pd.Series(dtype=float),
                gate_utc=T0 - pd.Timedelta(hours=13),
                book=PositionBook(),
            )

    def test_missing_clearing_price_rejects_order(self):
        reports = execute_auction_orders(
            market="ida2",
            session=self.session(),
            event_products=(QH,),
            orders={QH: 5.0},
            clearing_prices=pd.Series(dtype=float),
            gate_utc=T0 - pd.Timedelta(hours=13),
            book=PositionBook(),
        )
        assert reports[0].filled_mw == 0.0
        assert reports[0].reason == "no_clearing_price"

    def test_volume_cap(self):
        book = PositionBook()
        reports = execute_auction_orders(
            market="daa",
            session=self.session(max_volume_mw=15.0),
            event_products=(QH,),
            orders={QH: -40.0},
            clearing_prices=pd.Series({T0: 80.0}),
            gate_utc=T0 - pd.Timedelta(hours=13),
            book=book,
        )
        assert reports[0].filled_mw == -15.0
        assert reports[0].reason == "volume capped"
        assert book.net_position_mw(QH) == pytest.approx(-15.0)


class TestIdcExecution:
    def test_index_selection_by_lead(self):
        cfg = IdcConfig()
        assert select_idc_index(pd.Timedelta("45min"), cfg)[0] == "ID1"
        assert select_idc_index(pd.Timedelta("2h"), cfg)[0] == "ID3"
        assert select_idc_index(pd.Timedelta("8h"), cfg)[0] == "IDFULL"

    def test_execution_at_index_price_with_fallback(self):
        cfg = IdcConfig(transaction_cost_eur_per_mwh=0.4)
        idx = pd.DataFrame({"ID1": [float("nan")], "ID3": [55.0], "IDFULL": [52.0]}, index=[T0])
        book = PositionBook()
        reports = execute_idc_orders(
            cfg=cfg,
            event_products=(QH,),
            orders={QH: 8.0},
            decision_utc=T0 - pd.Timedelta(minutes=45),
            idc_indices=idx,
            book=book,
        )
        # lead 45 min prefers ID1, NaN -> falls back to ID3
        assert reports[0].price_eur_per_mwh == pytest.approx(55.0)
        assert reports[0].trades[0].transaction_cost_eur == pytest.approx(0.4 * 8.0 * 0.25)

    def test_no_price_rejects(self):
        reports = execute_idc_orders(
            cfg=IdcConfig(),
            event_products=(QH,),
            orders={QH: 8.0},
            decision_utc=T0 - pd.Timedelta(hours=1),
            idc_indices=pd.DataFrame(),
            book=PositionBook(),
        )
        assert reports[0].filled_mw == 0.0 and reports[0].reason == "no_price_data"

    def test_gate_enforced_via_eligibility(self):
        with pytest.raises(ValueError, match="not tradable"):
            execute_idc_orders(
                cfg=IdcConfig(),
                event_products=(),
                orders={QH: 5.0},
                decision_utc=T0 - pd.Timedelta(minutes=10),
                idc_indices=pd.DataFrame(),
                book=PositionBook(),
            )


class TestSettlement:
    def test_rebap_hand_calculated(self):
        rebap = pd.Series({T0: 120.0})
        s = ImbalanceSettlement(ImbalanceConfig(model="rebap"), rebap)
        # short 2 MWh at reBAP 120 -> pay 240
        r = s.settle(QH, delivered_mwh=3.0, contracted_mwh=5.0)
        assert r.deviation_mwh == pytest.approx(-2.0)
        assert r.cash_eur == pytest.approx(-240.0)
        # long 1 MWh at negative reBAP -> pay 50
        rebap[T0] = -50.0
        r = s.settle(QH, delivered_mwh=6.0, contracted_mwh=5.0)
        assert r.cash_eur == pytest.approx(-50.0)

    def test_deviation_penalty_added_once(self):
        s = ImbalanceSettlement(
            ImbalanceConfig(model="rebap", deviation_penalty_eur_per_mwh=10.0),
            pd.Series({T0: 100.0}),
        )
        r = s.settle(QH, 3.0, 5.0)
        assert r.cash_eur == pytest.approx(-200.0)
        assert r.penalty_eur == pytest.approx(-20.0)

    def test_asymmetric_spread(self):
        ref = pd.Series({T0: 100.0})
        s = ImbalanceSettlement(
            ImbalanceConfig(model="asymmetric_spread", spread_eur_per_mwh=25.0),
            rebap=pd.Series(dtype=float),
            reference_prices=ref,
        )
        long = s.settle(QH, 6.0, 5.0)  # surplus sold at 75
        short = s.settle(QH, 4.0, 5.0)  # shortfall bought at 125
        assert long.cash_eur == pytest.approx(+75.0)
        assert short.cash_eur == pytest.approx(-125.0)

    def test_missing_rebap_raises(self):
        s = ImbalanceSettlement(ImbalanceConfig(model="rebap"), pd.Series(dtype=float))
        with pytest.raises(KeyError):
            s.settle(QH, 1.0, 1.0)
