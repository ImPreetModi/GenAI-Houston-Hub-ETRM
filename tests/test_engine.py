"""Run with `pytest` (or `python tests/test_engine.py` - plain asserts, no pytest needed)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etrm.market_data import TZ, synthetic_hub_prices
from etrm.trade import Trade, generate_delivery_records
from etrm.portfolio import combine_deliveries, net_position
from etrm.pnl import Scenario, mark_to_market, apply_scenario
from etrm.pipeline import run_pipeline
from etrm.copilot import build_context, rule_based_explanation


def flat_prices(start, end, level):
    idx = pd.date_range(start, end, freq="h", tz=TZ, inclusive="left")
    return pd.Series(float(level), index=idx)


def T(**kw):
    base = dict(trade_id="T1", counterparty="X", side="BUY", profile="ATC",
                mw=10, price=40, start="2026-12-01", end="2027-01-01")
    base.update(kw)
    return Trade(**base)


def test_atc_hours_december():
    assert len(generate_delivery_records(T())) == 31 * 24


def test_on_peak_hours_december():
    # Dec 2026 has 23 weekdays x 16 on-peak hours (HE07-22); NERC holidays ignored
    assert len(generate_delivery_records(T(profile="ON_PEAK"))) == 23 * 16


def test_peak_plus_offpeak_equals_atc():
    n = len(generate_delivery_records(T(profile="ON_PEAK"))) + len(generate_delivery_records(T(profile="OFF_PEAK")))
    assert n == len(generate_delivery_records(T()))


def test_dst_spring_forward_has_743_hours():
    assert len(generate_delivery_records(T(start="2027-03-01", end="2027-04-01"))) == 31 * 24 - 1


def test_dst_fall_back_has_25_hour_day():
    assert len(generate_delivery_records(T(start="2026-11-01", end="2026-11-02"))) == 25


def test_net_position_netting():
    buy = generate_delivery_records(T(trade_id="B", mw=50))
    sell = generate_delivery_records(T(trade_id="S", side="SELL", profile="ON_PEAK", mw=30))
    net = net_position(combine_deliveries([buy, sell]))
    on_peak = (net.index.dayofweek < 5) & (net["hour_ending"] >= 7) & (net["hour_ending"] <= 22)
    assert (net.loc[on_peak, "net_mwh"] == 20).all()
    assert (net.loc[~on_peak, "net_mwh"] == 50).all()


def test_mtm_sign_buy_and_sell():
    p = flat_prices("2026-12-01", "2026-12-02", 50)
    buy = mark_to_market(generate_delivery_records(T(start="2026-12-01", end="2026-12-02")), p)
    sell = mark_to_market(generate_delivery_records(T(side="SELL", start="2026-12-01", end="2026-12-02")), p)
    assert buy["mtm_pnl"].sum() == 24 * 10 * (50 - 40)
    assert sell["mtm_pnl"].sum() == -24 * 10 * (50 - 40)


def test_stress_pct_shift():
    p = flat_prices("2026-12-01", "2026-12-02", 50)
    shocked = apply_scenario(p, Scenario("up", pct_shift=10))
    assert abs(shocked - 55).max() < 1e-9
    d = generate_delivery_records(T(start="2026-12-01", end="2026-12-02"))
    assert abs(mark_to_market(d, shocked)["mtm_pnl"].sum() - 24 * 10 * (55 - 40)) < 1e-6


def test_evening_shock_only_hits_he17_21():
    p = flat_prices("2026-12-01", "2026-12-02", 50)
    s = apply_scenario(p, Scenario("eve", peak_pct_shift=100))
    he = s.index.hour + 1
    assert abs(s[(he >= 17) & (he <= 21)] - 100).max() < 1e-9
    assert abs(s[~((he >= 17) & (he <= 21))] - 50).max() < 1e-9


def test_missing_prices_raise():
    p = flat_prices("2026-12-01", "2026-12-02", 50)
    d = generate_delivery_records(T(start="2026-12-01", end="2026-12-03"))
    try:
        mark_to_market(d, p)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pipeline_and_copilot_fallback():
    trades = [T(trade_id="BUY-1", mw=50, price=38), T(trade_id="SELL-1", side="SELL", profile="ON_PEAK", mw=30, price=45)]
    prices = synthetic_hub_prices("2026-12-01", "2027-01-01")
    sc = Scenario("Cold snap", pct_shift=30, abs_shift=25)
    res = run_pipeline(trades, prices, sc)
    assert abs(res["by_trade"]["delta"].sum() - (res["stress_total"] - res["base_total"])) < 1e-6
    ctx = build_context(res)
    text = rule_based_explanation(ctx)
    assert "MtM" in text and len(text) > 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
