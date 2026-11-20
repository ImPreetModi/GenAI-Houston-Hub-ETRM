"""End-to-end run shared by the Streamlit app, the API and the tests."""
from __future__ import annotations

import pandas as pd

from .trade import Trade, generate_delivery_records
from .portfolio import combine_deliveries, net_position, position_summary
from .pnl import Scenario, mark_to_market, apply_scenario, run_scenarios, PRESETS


def run_pipeline(trades: list[Trade], prices: pd.Series, stress: Scenario, price_meta: dict | None = None) -> dict:
    deliveries = combine_deliveries([generate_delivery_records(t) for t in trades])
    net = net_position(deliveries)
    base = mark_to_market(deliveries, prices)
    stressed = mark_to_market(deliveries, apply_scenario(prices, stress))
    stressed["pnl_delta"] = stressed["mtm_pnl"] - base["mtm_pnl"]

    by_trade = (
        pd.DataFrame(
            {
                "base_mtm": base.groupby("trade_id")["mtm_pnl"].sum(),
                "stress_mtm": stressed.groupby("trade_id")["mtm_pnl"].sum(),
            }
        )
        .assign(delta=lambda d: d["stress_mtm"] - d["base_mtm"])
        .reset_index()
    )
    scenarios = [s for s in PRESETS if (s.pct_shift, s.abs_shift, s.peak_pct_shift) != (stress.pct_shift, stress.abs_shift, stress.peak_pct_shift)]
    scenarios.append(stress)
    return {
        "trades": trades,
        "deliveries": deliveries,
        "net": net,
        "position": position_summary(net),
        "mtm_base": base,
        "mtm_stress": stressed,
        "by_trade": by_trade,
        "scenario_table": run_scenarios(deliveries, prices, scenarios),
        "stress": stress,
        "price_meta": price_meta or {"source": "unspecified"},
        "base_total": float(base["mtm_pnl"].sum()),
        "stress_total": float(stressed["mtm_pnl"].sum()),
    }
