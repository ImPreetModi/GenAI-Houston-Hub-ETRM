"""Physical power trade capture and hourly delivery record generation."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .market_data import TZ, HUB

PROFILES = ("ATC", "ON_PEAK", "OFF_PEAK")


@dataclass(frozen=True)
class Trade:
    trade_id: str
    counterparty: str
    side: str            # "BUY" (we receive power) or "SELL" (we deliver power)
    profile: str         # ATC | ON_PEAK | OFF_PEAK
    mw: float            # flat MW in every delivery hour
    price: float         # fixed $/MWh
    start: str           # first delivery day, inclusive (YYYY-MM-DD)
    end: str             # last delivery day, EXCLUSIVE
    hub: str = HUB

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1

    def validate(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if not (self.mw > 0 and math.isfinite(self.mw)):
            raise ValueError("mw must be a positive number")
        if not math.isfinite(self.price):
            raise ValueError("price must be finite")
        if pd.Timestamp(self.end) <= pd.Timestamp(self.start):
            raise ValueError("end must be after start")

    def to_dict(self) -> dict:
        return asdict(self)


def profile_mask(idx: pd.DatetimeIndex, profile: str) -> np.ndarray:
    """ERCOT-style blocks: on-peak = Mon-Fri HE07-HE22 (NERC holidays ignored here)."""
    he = idx.hour.to_numpy() + 1  # hour ending = hour beginning + 1
    on_peak = (idx.dayofweek.to_numpy() < 5) & (he >= 7) & (he <= 22)
    if profile == "ATC":
        return np.ones(len(idx), dtype=bool)
    return on_peak if profile == "ON_PEAK" else ~on_peak


def generate_delivery_records(trade: Trade) -> pd.DataFrame:
    """One row per delivery hour. Uses a tz-aware range so DST days have 23/25 hours."""
    trade.validate()
    idx = pd.date_range(trade.start, trade.end, freq="h", tz=TZ, inclusive="left")
    idx = idx[profile_mask(idx, trade.profile)]
    df = pd.DataFrame(
        {
            "trade_id": trade.trade_id,
            "counterparty": trade.counterparty,
            "side": trade.side,
            "hub": trade.hub,
            "profile": trade.profile,
            "delivery_start": idx,
            "hour_ending": idx.hour + 1,
            "mw": trade.mw,
            "mwh": trade.mw * 1.0,
            "trade_price": trade.price,
        }
    )
    df["signed_mwh"] = df["mwh"] * trade.sign
    df["contract_value"] = df["signed_mwh"] * df["trade_price"]
    return df.reset_index(drop=True)
