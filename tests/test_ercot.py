"""ERCOT client tests against a fake HTTP session (no network, no credentials needed)."""
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etrm import ercot_api
from etrm.ercot_api import ErcotAPIError, ErcotClient, parse_dam_spp
from etrm.price_curve import build_price_curve
from etrm.pnl import Scenario
from etrm.pipeline import run_pipeline
from etrm.trade import Trade
from etrm.copilot import build_context, rule_based_explanation

ercot_api.time.sleep = lambda *_: None  # no real sleeping in tests

FIELDS = [{"name": n} for n in ["deliveryDate", "hourEnding", "settlementPoint", "settlementPointPrice", "DSTFlag"]]


def make_rows(start, end, point="HB_HOUSTON", price_fn=lambda d, h: 30 + h):
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        n = 25 if d.strftime("%Y-%m-%d") == "2026-11-01" else 24
        for k in range(1, n + 1):
            he = k if (n == 24 or k <= 2) else k - 1
            flag = (n == 25 and k == 3)  # repeated HE02 is the standard-time hour
            rows.append([d.strftime("%Y-%m-%d"), f"{he:02d}:00", point, price_fn(d, k), flag])
            rows.append([d.strftime("%Y-%m-%d"), f"{he:02d}:00", "HB_NORTH", 999.0, flag])
    return rows


class Resp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = code, body or {}, headers or {}, str(body)

    def json(self):
        return self._b


class FakeSession:
    def __init__(self, rows, page_cap=500, reject_filter=False, fail_first=None):
        self.rows, self.page_cap, self.reject_filter = rows, page_cap, reject_filter
        self.fail_first = list(fail_first or [])
        self.token_calls = self.get_calls = 0

    def post(self, url, data=None, timeout=None):
        assert url == ercot_api.TOKEN_URL
        assert data["grant_type"] == "password" and data["client_id"] == ercot_api.CLIENT_ID
        self.token_calls += 1
        return Resp(200, {"id_token": f"tok{self.token_calls}", "access_token": "x", "expires_in": "3600"})

    def get(self, url, headers=None, params=None, timeout=None):
        self.get_calls += 1
        assert headers["Ocp-Apim-Subscription-Key"] == "KEY"
        assert headers["Authorization"].startswith("Bearer tok")
        if self.fail_first:
            return Resp(self.fail_first.pop(0))
        if "settlementPoint" in params and self.reject_filter:
            return Resp(400, {"message": "bad param"})
        d0, d1 = params["deliveryDateFrom"], params["deliveryDateTo"]
        rows = [r for r in self.rows if d0 <= r[0] <= d1]
        if "settlementPoint" in params:
            rows = [r for r in rows if r[2] == params["settlementPoint"]]
        size, page = params["size"], params["page"]
        cap = min(size, self.page_cap)
        pages = max(1, -(-len(rows) // cap))
        return Resp(200, {"_meta": {"totalPages": pages}, "fields": FIELDS, "data": rows[(page - 1) * cap: page * cap]})


def client(rows, **kw):
    return ErcotClient("u", "p", "KEY", session=FakeSession(rows, **kw))


def test_paging_and_filter():
    c = client(make_rows("2026-11-10", "2026-11-12"), page_cap=20)  # forces several pages
    s = c.dam_spp(date(2026, 11, 10), date(2026, 11, 12))
    assert len(s) == 72 and s.index.tz is not None and (s < 100).all()


def test_dst_fall_back_gives_25_unique_hours():
    c = client(make_rows("2026-11-01", "2026-11-01"))
    s = c.dam_spp(date(2026, 11, 1), date(2026, 11, 1))
    assert len(s) == 25 and s.index.is_unique


def test_400_on_filter_falls_back_to_local_filter():
    c = client(make_rows("2026-11-10", "2026-11-10"), reject_filter=True)
    s = c.dam_spp(date(2026, 11, 10), date(2026, 11, 10))
    assert len(s) == 24 and (s < 100).all()  # HB_NORTH (999) filtered out locally


def test_401_refreshes_token_once_and_429_retries():
    sess = FakeSession(make_rows("2026-11-10", "2026-11-10"), fail_first=[401, 429])
    c = ErcotClient("u", "p", "KEY", session=sess)
    assert len(c.dam_spp(date(2026, 11, 10), date(2026, 11, 10))) == 24
    assert sess.token_calls == 2  # initial + one refresh after the 401


def test_bad_credentials_message_does_not_leak_password():
    class Bad(FakeSession):
        def post(self, url, data=None, timeout=None):
            return Resp(400, {"error": "invalid_grant", "error_description": "AADB2C90225: wrong username or password"})
    c = ErcotClient("u", "SECRET-PW", "KEY", session=Bad([]))
    try:
        c._get_token()
        raise AssertionError("expected error")
    except ErcotAPIError as e:
        assert "SECRET-PW" not in str(e) and "AADB2C90225" in str(e)


def test_unexpected_columns_raise():
    try:
        parse_dam_spp(pd.DataFrame({"foo": [1]}))
        raise AssertionError("expected error")
    except ErcotAPIError:
        pass


def test_curve_actual_plus_proxy_and_scaling():
    # history Oct 1 - Nov 20; delivery Nov 19 - Dec 3 -> Nov 19-21 actual, rest proxy
    rows = make_rows("2026-10-01", "2026-11-21")
    c = client(rows)
    asof = pd.Timestamp("2026-11-20 10:00", tz="America/Chicago")
    prices, meta, is_proxy = build_price_curve(c, "2026-11-19", "2026-12-03", as_of=asof, lookback_days=45)
    assert not prices.isna().any() and len(prices) == 14 * 24
    assert meta["hours_dam_actual"] == 3 * 24 and meta["hours_forward_proxy"] == 11 * 24
    assert is_proxy.sum() == 11 * 24
    # scaled proxy mean equals requested forward ATC
    prices2, meta2, m2 = build_price_curve(c, "2026-11-19", "2026-12-03", as_of=asof, forward_atc=55.0)
    assert abs(prices2[m2].mean() - 55.0) < 1e-6
    assert "55.00" in meta2["proxy_method"]


def test_end_to_end_with_live_style_prices():
    c = client(make_rows("2026-10-01", "2026-11-21"))
    asof = pd.Timestamp("2026-11-20 10:00", tz="America/Chicago")
    prices, meta, _ = build_price_curve(c, "2026-12-01", "2027-01-01", as_of=asof)
    t = [Trade("BUY-1", "A", "BUY", "ATC", 50, 38, "2026-12-01", "2027-01-01")]
    res = run_pipeline(t, prices, Scenario("s", pct_shift=20), price_meta=meta)
    assert res["price_meta"]["hours_forward_proxy"] == 744
    assert "forward proxy" in rule_based_explanation(build_context(res))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} ERCOT tests passed")
