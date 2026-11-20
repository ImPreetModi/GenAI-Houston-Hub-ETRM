"""Build the hourly price curve for the delivery window from ERCOT data.

- Hours that already have published DAM prices use the actual HB_HOUSTON DAM SPP.
- Future hours have no price yet, so they use a FORWARD PROXY: the average hourly shape
  (weekday/weekend x hour) of the last N days of real DAM prices, optionally rescaled to a
  forward ATC level you enter (e.g. from your broker or ICE).
This is a stand-in for a proper forward curve and is labelled as such everywhere.
"""
from __future__ import annotations

import pandas as pd

from .ercot_api import ErcotAPIError, ErcotClient
from .market_data import HUB, TZ


def build_price_curve(client: ErcotClient, delivery_start: str, delivery_end: str,
                      as_of: pd.Timestamp | None = None, lookback_days: int = 45,
                      forward_atc: float | None = None, settlement_point: str = HUB):
    idx = pd.date_range(delivery_start, delivery_end, freq="h", tz=TZ, inclusive="left")
    today = (as_of or pd.Timestamp.now(tz=TZ)).date()
    last_dam_day = today + pd.Timedelta(days=1)  # tomorrow's DAM is posted the afternoon before
    d0 = idx[0].date()
    d1 = (idx[-1]).date()

    actual = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=TZ))
    if d0 <= last_dam_day:
        actual = client.dam_spp(d0, min(d1, last_dam_day), settlement_point)
    actual = actual.reindex(idx)
    missing = actual.isna()

    note = ""
    prices = actual.copy()
    if missing.any():
        hist = client.dam_spp(today - pd.Timedelta(days=lookback_days), today, settlement_point)
        if hist.empty:
            raise ErcotAPIError("No recent DAM prices returned; cannot build forward proxy")
        key = lambda ix: pd.MultiIndex.from_arrays([(ix.dayofweek >= 5), ix.hour])  # noqa: E731
        shape = hist.groupby([(hist.index.dayofweek >= 5), hist.index.hour]).mean()
        proxy = pd.Series(shape.reindex(key(idx[missing])).to_numpy(), index=idx[missing])
        if proxy.isna().any():
            proxy = proxy.fillna(hist.mean())
        if forward_atc:
            proxy = proxy * (forward_atc / proxy.mean())
        prices[missing] = proxy
        if d1 < today:
            note = "Some past delivery hours had no DAM data returned (very old dates may need ERCOT's archive endpoint); those hours use the proxy."

    prices.name = "price_usd_mwh"
    n_proxy = int(missing.sum())
    meta = {
        "source": "ERCOT Public API, NP4-190-CD DAM SPP, " + settlement_point,
        "hours_total": int(len(idx)),
        "hours_dam_actual": int(len(idx) - n_proxy),
        "hours_forward_proxy": n_proxy,
        "proxy_method": (f"avg weekday/weekend x hour shape of last {lookback_days} days of DAM"
                         + (f", rescaled to ATC {forward_atc:.2f} $/MWh" if forward_atc else "")) if n_proxy else "n/a",
        "note": note,
    }
    return prices, meta, missing
