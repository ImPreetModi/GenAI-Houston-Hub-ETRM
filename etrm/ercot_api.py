"""Client for the ERCOT Public Data API (api.ercot.com).

Auth (per ERCOT's developer docs): every call needs TWO credentials
  1. an Azure B2C ID token  -> POST portal username/password to the B2C token URL (valid 1 hour)
  2. your subscription key  -> 'Ocp-Apim-Subscription-Key' header

Env vars: ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY
Smoke test: python -m etrm.ercot_api
"""
from __future__ import annotations

import os
import re
import time
from datetime import date

import pandas as pd
import requests

from .market_data import HUB, TZ

TOKEN_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"  # public value from ERCOT's docs, not a secret
BASE_URL = "https://api.ercot.com/api/public-reports"
DAM_SPP_ENDPOINT = "/np4-190-cd/dam_stlmnt_pnt_prices"  # NP4-190-CD DAM Settlement Point Prices
TOKEN_TTL_SECONDS = 3300  # tokens last 3600s; refresh a bit early
TIMEOUT = (10, 30)


class ErcotAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class ErcotClient:
    def __init__(self, username: str, password: str, subscription_key: str,
                 session=None, sleep_seconds: float = 0.3, max_retries: int = 4):
        if not (username and password and subscription_key):
            raise ErcotAPIError("Missing ERCOT username, password or subscription key")
        self._username, self._password, self._key = username, password, subscription_key
        self.session = session or requests.Session()
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self._token: str | None = None
        self._token_time = 0.0

    @classmethod
    def from_env(cls, **kw) -> "ErcotClient":
        return cls(os.getenv("ERCOT_USERNAME", ""), os.getenv("ERCOT_PASSWORD", ""),
                   os.getenv("ERCOT_SUBSCRIPTION_KEY", ""), **kw)

    # ---------------------------------------------------------------- auth
    def _get_token(self, force: bool = False) -> str:
        if not force and self._token and time.time() - self._token_time < TOKEN_TTL_SECONDS:
            return self._token
        payload = {
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
            "response_type": "id_token",
            "scope": f"openid {CLIENT_ID} offline_access",
            "client_id": CLIENT_ID,
        }
        resp = self.session.post(TOKEN_URL, data=payload, timeout=TIMEOUT)
        try:
            body = resp.json()
        except Exception:
            raise ErcotAPIError(f"Token endpoint returned non-JSON (HTTP {resp.status_code})", resp.status_code)
        tok = body.get("id_token")
        if not tok:
            # never echo the password; B2C returns a safe error_description
            raise ErcotAPIError(f"Could not get ERCOT token: {body.get('error_description') or body.get('error') or 'unknown error'}",
                                resp.status_code)
        self._token, self._token_time = tok, time.time()
        return tok

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}", "Ocp-Apim-Subscription-Key": self._key,
                "Accept": "application/json"}

    # ---------------------------------------------------------------- http
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = BASE_URL + path
        delay, refreshed = 1.0, False
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=TIMEOUT)
            code = resp.status_code
            if code == 200:
                return resp.json()
            if code == 401 and not refreshed:  # token expired/invalid -> mint a new one once
                self._get_token(force=True)
                refreshed = True
                continue
            if code in (429, 502, 503, 504) and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                delay *= 2
                continue
            raise ErcotAPIError(f"ERCOT API {code} for {path}: {str(getattr(resp, 'text', ''))[:200]}", code)
        raise ErcotAPIError(f"ERCOT API retries exhausted for {path}")

    def _paged(self, path: str, params: dict, page_size: int = 10000) -> pd.DataFrame:
        params = {**params, "size": page_size, "page": 1}
        first = self._get(path, params)
        fields = [f["name"] for f in first.get("fields", [])]
        rows = list(first.get("data", []))
        total_pages = int(first.get("_meta", {}).get("totalPages", 1) or 1)
        for page in range(2, total_pages + 1):
            time.sleep(self.sleep_seconds)
            rows.extend(self._get(path, {**params, "page": page}).get("data", []))
        return pd.DataFrame(rows, columns=fields) if fields else pd.DataFrame()

    # ---------------------------------------------------------------- data
    def dam_spp(self, start: date, end: date, settlement_point: str = HUB) -> pd.Series:
        """Hourly DAM settlement point price ($/MWh) for delivery dates start..end INCLUSIVE.
        Index = tz-aware (America/Chicago) hour-beginning timestamps."""
        params = {"deliveryDateFrom": start.isoformat(), "deliveryDateTo": end.isoformat(),
                  "settlementPoint": settlement_point}
        try:
            df = self._paged(DAM_SPP_ENDPOINT, params)
        except ErcotAPIError as e:
            if e.status != 400:
                raise
            # server rejected the server-side filter: retry unfiltered and filter locally
            params.pop("settlementPoint")
            df = self._paged(DAM_SPP_ENDPOINT, params)
        return parse_dam_spp(df, settlement_point)


def parse_dam_spp(df: pd.DataFrame, settlement_point: str = HUB) -> pd.Series:
    """Turn the API's rows into an hourly tz-aware price Series (handles DST via DSTFlag)."""
    empty = pd.Series(dtype=float, name="price_usd_mwh", index=pd.DatetimeIndex([], tz=TZ))
    if df is None or df.empty:
        return empty
    cols = {_norm(c): c for c in df.columns}
    need = ["deliverydate", "hourending", "settlementpointprice"]
    if any(n not in cols for n in need):
        raise ErcotAPIError(f"Unexpected DAM SPP columns: {list(df.columns)}")
    if "settlementpoint" in cols:
        df = df[df[cols["settlementpoint"]].astype(str).str.strip() == settlement_point]
        if df.empty:
            return empty

    he = df[cols["hourending"]].astype(str).str.extract(r"(\d+)")[0].astype(int)
    naive = pd.DatetimeIndex(pd.to_datetime(df[cols["deliverydate"]]) + pd.to_timedelta(he - 1, unit="h"))
    price = pd.to_numeric(df[cols["settlementpointprice"]], errors="coerce").to_numpy()

    if "dstflag" in cols:
        flag = df[cols["dstflag"]].astype(str).str.strip().str.lower().isin(["true", "y", "yes", "1"]).to_numpy()
        # DSTFlag marks the repeated (standard-time) hour on fall-back day; ambiguous=True means DST
        idx = naive.tz_localize(TZ, ambiguous=~flag, nonexistent="shift_forward")
    else:
        idx = naive.tz_localize(TZ, ambiguous=True, nonexistent="shift_forward")

    s = pd.Series(price, index=idx, name="price_usd_mwh").dropna().sort_index()
    return s[~s.index.duplicated(keep="first")]


def main() -> None:  # smoke test: python -m etrm.ercot_api
    c = ErcotClient.from_env()
    print("1/3 Getting ID token ...")
    c._get_token()
    print("    OK")
    print("2/3 Listing public reports ...")
    r = c._get("")
    print("    OK, top-level keys:", list(r.keys())[:6])
    print("3/3 Fetching HB_HOUSTON DAM prices for the last 2 days ...")
    today = pd.Timestamp.now(tz=TZ).date()
    s = c.dam_spp(today - pd.Timedelta(days=2), today)
    print(f"    {len(s)} hourly rows")
    print(s.tail(6))


if __name__ == "__main__":
    main()
