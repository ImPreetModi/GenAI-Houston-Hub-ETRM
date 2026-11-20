"""Hourly price loading for ERCOT HB_HOUSTON (real CSV import or synthetic fallback)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/Chicago"  # ERCOT operates on Central Prevailing Time
HUB = "HB_HOUSTON"


def synthetic_hub_prices(start: str, end: str, seed: int = 7) -> pd.Series:
    """Realistic-looking hourly $/MWh series (morning + evening ramps, weekend
    discount, AR(1) noise, rare scarcity spikes). Interval is [start, end)."""
    idx = pd.date_range(start, end, freq="h", tz=TZ, inclusive="left")
    rng = np.random.default_rng(seed)
    hod = idx.hour.to_numpy()
    weekend = (idx.dayofweek.to_numpy() >= 5).astype(float)

    shape = (
        5 * np.exp(-((hod - 7) ** 2) / 8)
        + 12 * np.exp(-((hod - 19) ** 2) / 10)
        - 4 * np.exp(-((hod - 3) ** 2) / 18)
    )
    eps = rng.normal(0, 3, len(idx))
    ar = np.zeros(len(idx))
    for i in range(1, len(idx)):
        ar[i] = 0.7 * ar[i - 1] + eps[i]
    spikes = (rng.random(len(idx)) < 0.01) * rng.exponential(120, len(idx))

    price = np.clip(34 + shape - 3 * weekend + ar + spikes, -10, 5000)
    return pd.Series(price.round(2), index=idx, name="price_usd_mwh")


def load_prices_csv(file_or_path) -> pd.Series:
    """Load hourly prices from CSV. Two layouts are accepted:

    1. ERCOT DAM Settlement Point Prices export:
       Delivery Date, Hour Ending, Settlement Point, Settlement Point Price
       (filtered to HB_HOUSTON). Check column names against the file you download.
    2. Simple: delivery_start (ISO timestamp), price
    """
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if {"delivery_start", "price"} <= set(df.columns):
        idx = pd.to_datetime(df["delivery_start"], utc=True).dt.tz_convert(TZ)
        s = pd.Series(df["price"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx))
    elif {"delivery_date", "hour_ending", "settlement_point_price"} <= set(df.columns):
        if "settlement_point" in df.columns:
            df = df[df["settlement_point"] == HUB]
        if df.empty:
            raise ValueError(f"No rows for settlement point {HUB}")
        he = df["hour_ending"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        naive = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(he - 1, unit="h")
        naive = pd.DatetimeIndex(naive)
        try:  # ERCOT repeats HE2 on the fall-back day
            idx = naive.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            idx = naive.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
        s = pd.Series(df["settlement_point_price"].to_numpy(dtype=float), index=idx)
        s = s[~s.index.isna()]
    else:
        raise ValueError(
            "Unrecognised columns. Need (delivery_start, price) or "
            "(Delivery Date, Hour Ending, Settlement Point Price)."
        )

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s.name = "price_usd_mwh"
    return s
