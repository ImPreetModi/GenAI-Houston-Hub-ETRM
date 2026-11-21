"""Build the hourly price curve for the delivery window from ERCOT data.

- Hours that already have published DAM prices use the actual HB_HOUSTON DAM SPP.
- Future hours have no price yet, so they use a FORWARD PROXY: the average hourly shape
  (weekday/weekend x hour) of the last N days of real DAM prices, optionally rescaled to a
  forward ATC level you enter (e.g. from your broker or ICE).
This is a stand-in for a proper forward curve and is labelled as such everywhere.
"""
from __future__ import annotations

from datetime import timedelta

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


def dam_history(client: ErcotClient, days: int = 90, end=None, settlement_point: str = HUB,
                chunk_days: int = 30) -> pd.Series:
    """Hourly HB_HOUSTON DAM prices for the last `days` days (ending today), fetched in chunks.

    Walks backwards chunk by chunk. If ERCOT has no data (or rejects the request) for an older
    chunk, stops and returns what it has, so the caller can see how many days it actually got.
    """
    end_day = (end or pd.Timestamp.now(tz=TZ)).date()
    pieces, remaining, cur_end = [], int(days), end_day
    while remaining > 0:
        n = min(chunk_days, remaining)
        cur_start = cur_end - timedelta(days=n - 1)
        try:
            s = client.dam_spp(cur_start, cur_end, settlement_point)
        except ErcotAPIError as e:
            if pieces and e.status in (400, 404):
                break  # older data not available through this endpoint
            raise
        if s.empty and pieces:
            break
        if not s.empty:
            pieces.append(s)
        cur_end = cur_start - timedelta(days=1)
        remaining -= n
    if not pieces:
        raise ErcotAPIError("ERCOT returned no DAM price history for the requested window")
    out = pd.concat(pieces).sort_index()
    return out[~out.index.duplicated(keep="first")]


def history_to_curve(prices: pd.Series) -> pd.DataFrame:
    """Hourly series -> (date x hour-ending 1..24) matrix, the shape the VaR code expects."""
    d = pd.DataFrame({"date": prices.index.date, "he": prices.index.hour + 1, "price": prices.to_numpy()})
    curve = (d.pivot_table(index="date", columns="he", values="price", aggfunc="mean")
              .sort_index().reindex(columns=range(1, 25)))
    return curve.dropna(thresh=20).ffill(axis=1).bfill(axis=1)