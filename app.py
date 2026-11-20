"""Streamlit front end: trade -> deliveries -> net position -> MtM -> stress -> AI copilot.
Prices come live from the ERCOT Public API (HB_HOUSTON DAM SPP), with a synthetic offline fallback."""
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from etrm.copilot import build_context, explain
from etrm.ercot_api import ErcotAPIError, ErcotClient
from etrm.market_data import synthetic_hub_prices
from etrm.pipeline import run_pipeline
from etrm.pnl import Scenario
from etrm.price_curve import build_price_curve
from etrm.trade import PROFILES, Trade

st.set_page_config(page_title="Houston Hub Trading Copilot", page_icon="⚡", layout="wide")


def secret(name: str):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_client(username: str, password: str, key: str) -> ErcotClient:
    return ErcotClient(username, password, key)  # reused across reruns so the ID token is cached


@st.cache_data(ttl=3600, show_spinner="Pulling HB_HOUSTON prices from the ERCOT API...")
def get_live_curve(_client, start: str, end: str, lookback: int, fwd: float, today: str):
    return build_price_curve(_client, start, end, lookback_days=lookback, forward_atc=fwd or None)


@st.cache_data(show_spinner=False)
def get_synthetic(start: str, end: str) -> pd.Series:
    return synthetic_hub_prices(start, end)


def naive(s):
    s = s.copy()
    s.index = s.index.tz_localize(None)
    return s


def usd(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Trade ticket")
    counterparty = st.text_input("Counterparty", "Counterparty A")
    side = st.selectbox("Side", ["BUY", "SELL"])
    profile = st.selectbox("Profile", PROFILES, help="On-peak = Mon-Fri HE07-HE22")
    mw = st.number_input("Volume (MW, flat)", min_value=1.0, value=50.0, step=5.0)
    price = st.number_input("Contract price ($/MWh)", value=38.0, step=0.5)
    start = st.date_input("Delivery start", date(2026, 12, 1))
    end = st.date_input("Delivery end (exclusive)", date(2027, 1, 1))
    add_book = st.checkbox("Include existing book (SELL 30 MW on-peak @ $45)", True)

    st.header("2 · Market data")
    source = st.radio("HB_HOUSTON prices", ["ERCOT API (live)", "Synthetic demo (offline)"])
    creds = None
    lookback, fwd = 45, 0.0
    if source == "ERCOT API (live)":
        lookback = st.slider("Forward-proxy lookback (days)", 14, 90, 45,
                             help="Future hours are priced with the average hourly shape of this many recent days of real DAM prices.")
        fwd = st.number_input("Forward ATC override ($/MWh, 0 = none)", min_value=0.0, value=0.0, step=0.5,
                              help="Optional: rescale the proxy to a forward level you trust (e.g. from a broker).")
        u, p, k = secret("ERCOT_USERNAME"), secret("ERCOT_PASSWORD"), secret("ERCOT_SUBSCRIPTION_KEY")
        if not (u and p and k):
            with st.expander("ERCOT API credentials", expanded=True):
                st.caption("Or set ERCOT_USERNAME / ERCOT_PASSWORD / ERCOT_SUBSCRIPTION_KEY as env vars or Streamlit secrets. Not stored.")
                u = st.text_input("Portal username (email)")
                p = st.text_input("Portal password", type="password")
                k = st.text_input("Subscription key", type="password")
        creds = (u, p, k) if (u and p and k) else None

    st.header("3 · Stress scenario")
    pct = st.slider("All hours: price change (%)", -50, 100, 30)
    abs_shift = st.slider("All hours: add ($/MWh)", -20, 100, 25)
    peak = st.slider("Extra % on evening ramp (HE17-21)", 0, 300, 0)

st.title("⚡ Houston Hub Power Trading Copilot")
st.caption("ERCOT HB_HOUSTON · physical purchase → hourly delivery → net position → MtM → stress → AI explanation")

# ------------------------------------------------------------------ build + run
trades = [Trade("BUY-001" if side == "BUY" else "SELL-001", counterparty, side, profile, mw, price, str(start), str(end))]
if add_book:
    trades.append(Trade("SELL-RETAIL", "Retail Book", "SELL", "ON_PEAK", 30.0, 45.0, str(start), str(end)))

try:
    for t in trades:
        t.validate()
    if source == "ERCOT API (live)":
        if creds is None:
            st.info("Enter your ERCOT API credentials in the sidebar (or switch to the synthetic demo).")
            st.stop()
        prices, meta, is_proxy = get_live_curve(get_client(*creds), str(start), str(end), lookback, fwd,
                                                str(date.today()))
    else:
        prices = get_synthetic(str(start), str(end))
        meta = {"source": "Synthetic demo data (not real ERCOT prices)"}
        is_proxy = None
    stress = Scenario("Custom stress", pct_shift=pct, abs_shift=abs_shift, peak_pct_shift=peak)
    res = run_pipeline(trades, prices, stress, price_meta=meta)
except ErcotAPIError as e:
    st.error(f"ERCOT API problem: {e}")
    st.stop()
except requests.RequestException as e:
    st.error(f"Network problem reaching ERCOT: {type(e).__name__}")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

if source == "ERCOT API (live)":
    st.success(
        f"Live ERCOT data: {meta['hours_dam_actual']:,} of {meta['hours_total']:,} delivery hours use published DAM prices; "
        f"{meta['hours_forward_proxy']:,} use a forward proxy ({meta['proxy_method']}).",
        icon="✅",
    )
    if meta.get("note"):
        st.warning(meta["note"])
    if meta["hours_forward_proxy"]:
        st.caption("The forward proxy is NOT a market forward curve. Treat future-month MtM as indicative.")
else:
    st.warning("Synthetic demo prices, not real ERCOT data.", icon="⚠️")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net volume (MWh)", f"{res['position']['net_mwh_total']:,.0f}")
c2.metric("Base MtM", usd(res["base_total"]))
c3.metric("Stressed MtM", usd(res["stress_total"]), usd(res["stress_total"] - res["base_total"]))
c4.metric("Delivery hours", f"{res['position']['hours']:,}")

tabs = st.tabs(["1 Trade", "2 Deliveries", "3 Net position", "4 MtM P&L", "5 Stress", "6 AI copilot"])

with tabs[0]:
    st.subheader("Trade capture")
    st.dataframe(pd.DataFrame([t.to_dict() for t in res["trades"]]), hide_index=True)
    st.caption("Sign convention: BUY = long (+MWh), SELL = short (-MWh).")

with tabs[1]:
    st.subheader("Hourly delivery records")
    d = res["deliveries"]
    st.dataframe(d, hide_index=True)
    st.download_button("Download CSV", d.to_csv(index=False), "delivery_records.csv", "text/csv")

with tabs[2]:
    st.subheader("Net portfolio position (MWh per hour)")
    st.line_chart(naive(res["net"]["net_mwh"]))
    st.json(res["position"], expanded=False)
    st.dataframe(res["net"])

with tabs[3]:
    st.subheader("Market price curve used for MtM ($/MWh)")
    curve = pd.DataFrame({"price": prices})
    if is_proxy is not None:
        curve["DAM actual"] = prices.where(~is_proxy)
        curve["Forward proxy"] = prices.where(is_proxy)
        curve = curve[["DAM actual", "Forward proxy"]]
    curve.index = curve.index.tz_localize(None)
    st.line_chart(curve)
    st.subheader("Mark-to-market P&L")
    base = res["mtm_base"]
    st.caption("MtM = signed MWh × (market price − contract price), summed over delivery hours.")
    st.bar_chart(base.groupby(base["delivery_start"].dt.date)["mtm_pnl"].sum())
    st.dataframe(res["by_trade"], hide_index=True)
    st.dataframe(base[["trade_id", "delivery_start", "signed_mwh", "trade_price", "market_price", "mtm_pnl"]], hide_index=True)

with tabs[4]:
    st.subheader("Price stress")
    st.write(
        f"Scenario: **{pct:+d}%** on all hours, **{abs_shift:+d} $/MWh**, extra **{peak:+d}%** on HE17-21 → "
        f"P&L change **{usd(res['stress_total'] - res['base_total'])}**"
    )
    both = pd.DataFrame(
        {
            "base price": res["mtm_base"].groupby("delivery_start")["market_price"].mean(),
            "stressed price": res["mtm_stress"].groupby("delivery_start")["market_price"].mean(),
        }
    )
    both.index = both.index.tz_localize(None)
    st.line_chart(both)
    st.dataframe(res["scenario_table"], hide_index=True)
    st.bar_chart(res["scenario_table"].set_index("scenario")["delta_vs_base_usd"])

with tabs[5]:
    st.subheader("AI copilot")
    ctx = build_context(res)
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        key = st.text_input("Anthropic API key (optional, not stored)", type="password") or None
        st.caption("No key → a deterministic rule-based explainer is used. Add a key for a Claude-written explanation.")
    q = st.text_input("Ask a question (optional)", placeholder="Why does the SELL trade lose money in this scenario?")
    if st.button("Explain the result", type="primary"):
        with st.spinner("Thinking..."):
            text, src = explain(ctx, question=q or None, api_key=key)
        st.markdown(text)
        st.caption(f"Source: {src}")
    with st.expander("Context sent to the model (the model never computes numbers)"):
        st.json(ctx)
