"""Mark-to-market P&L and price stress scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def mark_to_market(deliveries: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """MtM per hour = signed_mwh * (market_price - trade_price).
    BUY gains when market > contract price; SELL gains when market < contract price."""
    df = deliveries.copy()
    market = prices.reindex(df["delivery_start"])
    missing = int(market.isna().sum())
    if missing:
        raise ValueError(f"{missing} delivery hours have no market price - check price file coverage")
    df["market_price"] = market.to_numpy()
    df["mtm_pnl"] = df["signed_mwh"] * (df["market_price"] - df["trade_price"])
    return df


@dataclass(frozen=True)
class Scenario:
    name: str
    pct_shift: float = 0.0       # % change applied to every hour
    abs_shift: float = 0.0       # $/MWh added to every hour (after pct)
    peak_pct_shift: float = 0.0  # extra % on the evening ramp, HE17-HE21

    def is_base(self) -> bool:
        return self.pct_shift == self.abs_shift == self.peak_pct_shift == 0


PRESETS = [
    Scenario("Base"),
    Scenario("Prices -20%", pct_shift=-20),
    Scenario("Prices +20%", pct_shift=20),
    Scenario("Cold snap (+30%, +$25)", pct_shift=30, abs_shift=25),
    Scenario("Evening scarcity (+150% HE17-21)", peak_pct_shift=150),
]


def apply_scenario(prices: pd.Series, sc: Scenario) -> pd.Series:
    p = prices * (1 + sc.pct_shift / 100) + sc.abs_shift
    he = prices.index.hour + 1
    evening = (he >= 17) & (he <= 21)
    return p.where(~evening, p * (1 + sc.peak_pct_shift / 100))


def run_scenarios(deliveries: pd.DataFrame, prices: pd.Series, scenarios: list[Scenario]) -> pd.DataFrame:
    base = mark_to_market(deliveries, prices)["mtm_pnl"].sum()
    rows = []
    for sc in scenarios:
        total = mark_to_market(deliveries, apply_scenario(prices, sc))["mtm_pnl"].sum()
        rows.append({"scenario": sc.name, "total_mtm_usd": total, "delta_vs_base_usd": total - base})
    return pd.DataFrame(rows)
