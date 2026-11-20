"""Net portfolio position across all trades."""
from __future__ import annotations

import pandas as pd


def combine_deliveries(delivery_frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(delivery_frames, ignore_index=True).sort_values(
        ["delivery_start", "trade_id"], ignore_index=True
    )


def net_position(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Hourly net MWh (+ long / - short). Long = buys, short = sells."""
    d = deliveries.assign(
        _long=deliveries["signed_mwh"].clip(lower=0),
        _short=deliveries["signed_mwh"].clip(upper=0),
    )
    out = d.groupby("delivery_start").agg(
        long_mwh=("_long", "sum"),
        short_mwh=("_short", "sum"),
        net_mwh=("signed_mwh", "sum"),
    )
    out["hour_ending"] = out.index.hour + 1
    return out


def position_summary(net: pd.DataFrame) -> dict:
    return {
        "hours": int(len(net)),
        "net_mwh_total": float(net["net_mwh"].sum()),
        "max_long_mw": float(net["net_mwh"].max()),
        "min_net_mw": float(net["net_mwh"].min()),
        "hours_long": int((net["net_mwh"] > 0).sum()),
        "hours_short": int((net["net_mwh"] < 0).sum()),
        "hours_flat": int((net["net_mwh"] == 0).sum()),
    }
