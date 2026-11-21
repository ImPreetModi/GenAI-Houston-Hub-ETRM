
import os

import pandas as pd
import requests
import streamlit as st
import numpy as np


from etrm.copilot import build_context, explain
from etrm.ercot_api import ErcotAPIError, ErcotClient
from etrm.market_data import synthetic_hub_prices
from etrm.pipeline import run_pipeline
from etrm.pnl import Scenario
from etrm.price_curve import build_price_curve, dam_history, history_to_curve
from etrm.trade import PROFILES, Trade
from datetime import date, timedelta

from statistics import NormalDist

_norm = NormalDist()


# ============================================================
# VALUE AT RISK (inline)
# ============================================================

def load_dam_history(file, hub="HB_HOUSTON", min_days=30):
    df = pd.read_csv(file)
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise ValueError(f"CSV is missing a column like {names}")

    d = pick("date", "deliverydate", "operday")
    h = pick("hourending", "he", "hour")
    p = pick("price", "settlementpointprice", "spp", "lmp")
    sp = next((cols[n] for n in ("settlementpoint", "settlementpointname") if n in cols), None)
    if sp:
        df = df[df[sp].astype(str).str.upper() == hub.upper()]

    out = pd.DataFrame({
        "date": pd.to_datetime(df[d]).dt.date,
        "hour_ending": df[h].astype(str).str.split(":").str[0].astype(int),
        "price": pd.to_numeric(df[p], errors="coerce"),
    }).dropna()

    curve = (out.pivot_table(index="date", columns="hour_ending", values="price", aggfunc="mean")
                .sort_index().reindex(columns=range(1, 25)))
    curve = curve.dropna(thresh=20).ffill(axis=1).bfill(axis=1)
    if len(curve) < min_days:
        raise ValueError(f"Need at least {min_days} days of history, got {len(curve)}")
    return curve


def position_profile(net):
    s = net["net_mwh"] if isinstance(net, pd.DataFrame) else net
    he = pd.DatetimeIndex(s.index).hour + 1
    return s.groupby(he).mean().reindex(range(1, 25), fill_value=0.0).astype(float)


def var_summary(curve, pos, conf=0.95, horizon_days=1, n_sims=10_000, seed=42):
    dP = curve.diff().dropna()
    w = pos.values
    k = np.sqrt(horizon_days)

    # Historical
    hist_pnl = dP.values @ w
    hv = -np.percentile(hist_pnl, (1 - conf) * 100)
    he = -hist_pnl[hist_pnl <= -hv].mean()

    # Parametric
    mu = dP.mean().values @ w
    sigma = float(np.sqrt(w @ dP.cov().values @ w))
    z = _norm.inv_cdf(conf)
    pv = z * sigma - mu
    pe = sigma * _norm.pdf(z) / (1 - conf) - mu

    # Monte Carlo
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(dP.mean().values, dP.cov().values,
                                   size=n_sims, check_valid="ignore")
    mc_pnl = sims @ w
    mv = -np.percentile(mc_pnl, (1 - conf) * 100)
    me = -mc_pnl[mc_pnl <= -mv].mean()

    table = pd.DataFrame({
        "method": ["Historical", "Parametric", "Monte Carlo"],
        "var_usd": [hv * k, pv * k, mv * k],
        "es_usd": [he * k, pe * k, me * k],
    }).round(0)
    return {"table": table, "hist_pnl": hist_pnl, "days": len(curve)}

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GenAI Houston Hub | ETRM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080f1c;
    --panel: #101c2e;
    --panel2: #142238;
    --border: #233750;
    --cyan: #00c8e8;
    --blue: #3185f5;
    --text: #e7f0fa;
    --muted: #8da2b9;
    --green: #27d69b;
    --red: #ff667d;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0b1728;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #c4d3e3 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

p, label, span {
    font-family: 'Inter', sans-serif;
}

.eyebrow {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero {
    background: linear-gradient(110deg, #122942 0%, #102137 55%, #102d48 100%);
    border: 1px solid #26445f;
    border-radius: 15px;
    padding: 26px 30px;
    margin: 5px 0 25px 0;
}

.hero-title {
    color: #f1f7ff;
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

.hero-subtitle {
    color: #a6bdd2;
    font-size: 13px;
    margin-top: 8px;
}

.hero-tag {
    display: inline-block;
    background: #123c51;
    border: 1px solid #17627a;
    color: #50ddf2;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 700;
    padding: 5px 9px;
    letter-spacing: 1px;
    margin-top: 15px;
}

.kpi-card {
    background: linear-gradient(145deg, #14243a, #101b2d);
    border: 1px solid #263b54;
    border-radius: 12px;
    padding: 19px 20px;
    min-height: 112px;
}

.kpi-label {
    color: #91a8bf;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.kpi-value {
    color: #f0f6ff;
    font-family: 'DM Mono', monospace;
    font-size: 25px;
    font-weight: 500;
    margin-top: 13px;
}

.kpi-sub {
    color: #8198af;
    font-size: 11px;
    margin-top: 5px;
}

.section-heading {
    color: #eaf3ff;
    font-size: 17px;
    font-weight: 700;
    margin: 8px 0 5px 0;
}

.section-caption {
    color: #8298b0;
    font-size: 12px;
    margin-bottom: 16px;
}

div[data-testid="stTabs"] button {
    color: #93a9bf;
    font-weight: 600;
    font-size: 12px;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #32d7ee !important;
    border-bottom-color: #00c8e8 !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #00c8e8;
}

div[data-testid="stMetric"] {
    background: #122239;
    border: 1px solid #263d57;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #9bb0c6;
}

div[data-testid="stMetricValue"] {
    color: #edf6ff;
    font-family: 'DM Mono', monospace;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263d57;
    border-radius: 9px;
    overflow: hidden;
}

.stButton > button {
    background: linear-gradient(90deg, #087fae, #0873d4);
    color: white;
    border: 1px solid #159bc9;
    border-radius: 7px;
    font-weight: 700;
    min-height: 40px;
}

.stButton > button:hover {
    background: #119bc5;
    color: white;
    border-color: #42dafa;
}

.stDownloadButton > button {
    background: #142b42;
    color: #52d8ee;
    border: 1px solid #28627c;
    border-radius: 7px;
}

.stTextInput input, .stNumberInput input {
    background: #111f32;
    color: #eaf3ff;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #111f32;
}

hr {
    border-color: #233750;
}

div[data-testid="stAlert"] {
    border-radius: 9px;
}

div[data-testid="stExpander"] {
    border: 1px solid #263d57;
    border-radius: 9px;
}

.footer {
    color: #647d96;
    font-size: 11px;
    text-align: center;
    padding-top: 35px;
    border-top: 1px solid #203249;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def secret(name: str):
    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name)
        return value if value else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_client(username: str, password: str, key: str) -> ErcotClient:
    return ErcotClient(username, password, key)


@st.cache_data(ttl=3600, show_spinner="Pulling HB_HOUSTON prices from ERCOT...")
def get_live_curve(_client, start: str, end: str, lookback: int, fwd: float, today: str):
    return build_price_curve(
        _client, start, end,
        lookback_days=lookback,
        forward_atc=fwd or None
    )


@st.cache_data(show_spinner=False)
def get_synthetic(start: str, end: str) -> pd.Series:
    return synthetic_hub_prices(start, end)


@st.cache_data(ttl=3600, show_spinner="Pulling DAM price history from ERCOT...")
def get_history_live(_client, days: int, today: str):
    return dam_history(_client, days=days)


@st.cache_data(show_spinner=False)
def get_history_synth(days: int, today: str):
    end = pd.Timestamp(today)
    return synthetic_hub_prices((end - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                                (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))


def naive(s):
    s = s.copy()
    s.index = s.index.tz_localize(None)
    return s


def usd(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def section_title(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if description:
        st.markdown(
            f'<div class="section-caption">{description}</div>',
            unsafe_allow_html=True
        )


def kpi_card(label, value, subtitle):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR: TRADING CONFIGURATION
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 5px 0 22px 0;">
        <div style="font-size:27px;font-weight:800;color:#eef7ff;">
            ⚡ HOUSTON HUB
        </div>
        <div style="font-size:10px;color:#38d5ee;letter-spacing:2px;">
            ENERGY TRADING TERMINAL
        </div>
        <div style="font-size:11px;color:#8299b1;margin-top:7px;">
            GENAI POWERED · ERCOT MARKET
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 01 / Trade ticket")

    counterparty = st.text_input("Counterparty", "Counterparty A")

    side = st.selectbox("Trade side", ["BUY", "SELL"])

    profile = st.selectbox(
        "Delivery profile",
        PROFILES,
        help="On-peak = Mon-Fri HE07-HE22"
    )

    mw = st.number_input(
        "Volume (MW)",
        min_value=1.0,
        value=50.0,
        step=5.0
    )

    price = st.number_input(
        "Contract price ($/MWh)",
        value=38.0,
        step=0.5
    )
    
    today = date.today()

    start = st.date_input(
        "Delivery start",
        today
    )
    
    end = st.date_input(
        "Delivery end (exclusive)",
        today + timedelta(days=1)
    )

    add_book = st.checkbox(
        "Include existing book",
        True,
        help="SELL 30 MW on-peak @ $45/MWh"
    )

    st.divider()

    st.markdown("### 02 / Market data")

    source = st.radio(
        "HB_HOUSTON price source",
        ["ERCOT API (live)", "Synthetic demo (offline)"]
    )

    creds = None
    lookback, fwd = 45, 0.0

    if source == "ERCOT API (live)":

        lookback = st.slider(
            "Forward-proxy lookback (days)",
            14, 90, 45
        )

        fwd = st.number_input(
            "Forward ATC override ($/MWh)",
            min_value=0.0,
            value=0.0,
            step=0.5
        )

        u = secret("ERCOT_USERNAME")
        p = secret("ERCOT_PASSWORD")
        k = secret("ERCOT_SUBSCRIPTION_KEY")

        if not (u and p and k):
            with st.expander("ERCOT API credentials", expanded=True):
                st.caption("Credentials are not stored by this form.")

                u = st.text_input("Portal username")
                p = st.text_input("Portal password", type="password")
                k = st.text_input("Subscription key", type="password")

        creds = (u, p, k) if (u and p and k) else None

    st.divider()

    st.markdown("### 03 / Stress scenario")

    pct = st.slider(
        "All hours: price change (%)",
        -50, 100, 30
    )

    abs_shift = st.slider(
        "All hours: add ($/MWh)",
        -20, 100, 25
    )

    peak = st.slider(
        "Evening ramp: extra (%)",
        0, 300, 0
    )

    st.divider()
    


    st.divider()
    st.markdown("### 04 / Value at Risk")
    var_days = st.slider("History window (days)", 30, 120, 90,
                         help="Pulled automatically from the ERCOT API (live mode) or generated (demo mode).")
    var_conf = st.select_slider("Confidence", [0.90, 0.95, 0.99], value=0.95,
                                format_func=lambda x: f"{x:.0%}")
    var_horizon = st.number_input("Horizon (days)", min_value=1, max_value=10, value=1)

    st.divider()

    
    st.markdown(
    '<span style="font-size:13px; color:gray;">Developed by Preet Modi</span>'
    '&nbsp;&nbsp;&nbsp;'
    '<a href="https://www.linkedin.com/in/preetjmodi/" target="_blank">'
    '<img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" '
    'width="14" style="vertical-align:middle; filter:invert(50%);">'
    '</a>',
    unsafe_allow_html=True
)
# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">ERCOT / HB_HOUSTON / POWER MARKETS</div>
    <div class="hero-title">Houston Hub Trading Copilot</div>
    <div class="hero-subtitle">
        Commodity trading intelligence · Hourly delivery ·
        Portfolio valuation · Risk analytics · AI-powered insights
    </div>
    <div class="hero-tag">LIVE MARKET ANALYTICS</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA PIPELINE
# ============================================================

trades = [
    Trade(
        "BUY-001" if side == "BUY" else "SELL-001",
        counterparty,
        side,
        profile,
        mw,
        price,
        str(start),
        str(end)
    )
]

if add_book:
    trades.append(
        Trade(
            "SELL-RETAIL",
            "Retail Book",
            "SELL",
            "ON_PEAK",
            30.0,
            45.0,
            str(start),
            str(end)
        )
    )


try:
    for t in trades:
        t.validate()

    if source == "ERCOT API (live)":

        if creds is None:
            st.info(
                "Enter your ERCOT credentials in the sidebar "
                "or switch to synthetic demo mode."
            )
            st.stop()

        prices, meta, is_proxy = get_live_curve(
            get_client(*creds),
            str(start),
            str(end),
            lookback,
            fwd,
            str(date.today())
        )

    else:
        prices = get_synthetic(str(start), str(end))

        meta = {
            "source": "Synthetic demo data (not real ERCOT prices)"
        }

        is_proxy = None

    stress = Scenario(
        "Custom stress",
        pct_shift=pct,
        abs_shift=abs_shift,
        peak_pct_shift=peak
    )

    res = run_pipeline(
        trades,
        prices,
        stress,
        price_meta=meta
    )

except ErcotAPIError as e:
    st.error(f"ERCOT API problem: {e}")
    st.stop()

except requests.RequestException as e:
    st.error(f"Network problem reaching ERCOT: {type(e).__name__}")
    st.stop()

except ValueError as e:
    st.error(str(e))
    st.stop()


# ============================================================
# MARKET STATUS
# ============================================================

if source == "ERCOT API (live)":

    st.success(
        f"Live ERCOT data: {meta['hours_dam_actual']:,} of "
        f"{meta['hours_total']:,} delivery hours use published DAM prices; "
        f"{meta['hours_forward_proxy']:,} use a forward proxy "
        f"({meta['proxy_method']})."
    )

    if meta.get("note"):
        st.warning(meta["note"])

    if meta["hours_forward_proxy"]:
        st.caption(
            "Forward proxy is not an exchange-traded forward curve. "
            "Future-month MtM is indicative."
        )

else:
    st.warning(
        "DEMO MODE — Synthetic prices, not real ERCOT market data."
    )


# ============================================================
# PORTFOLIO KPI CARDS
# ============================================================

section_title(
    "Portfolio overview",
    "Live trade valuation and delivery exposure"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "Net volume",
        f"{res['position']['net_mwh_total']:,.0f}",
        "Total net MWh"
    )

with c2:
    kpi_card(
        "Base MtM",
        usd(res["base_total"]),
        "Current market valuation"
    )

with c3:
    kpi_card(
        "Stressed MtM",
        usd(res["stress_total"]),
        f"Change: {usd(res['stress_total'] - res['base_total'])}"
    )

with c4:
    kpi_card(
        "Delivery hours",
        f"{res['position']['hours']:,}",
        "Hourly settlement periods"
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# WORKSPACE NAVIGATION
# ============================================================

tabs = st.tabs([
    "01  Trade Capture",
    "02  Deliveries",
    "03  Net Position",
    "04  MtM & P&L",
    "05  Stress Testing",
    "06  VaR & Risk",
    "07  AI Copilot"
])


# ============================================================
# 01 TRADE CAPTURE
# ============================================================

with tabs[0]:

    section_title(
        "Trade capture",
        "Physical power trade details and contract information"
    )

    st.dataframe(
        pd.DataFrame([t.to_dict() for t in res["trades"]]),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Position convention: BUY = long (+MWh), "
        "SELL = short (-MWh)."
    )


# ============================================================
# 02 HOURLY DELIVERIES
# ============================================================

with tabs[1]:

    section_title(
        "Hourly delivery records",
        "Scheduled delivery volumes across the selected contract period"
    )

    d = res["deliveries"]

    st.dataframe(
        d,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "↓ Download delivery records CSV",
        d.to_csv(index=False),
        "delivery_records.csv",
        "text/csv",
        use_container_width=False
    )


# ============================================================
# 03 NET PORTFOLIO POSITION
# ============================================================

with tabs[2]:

    section_title(
        "Net portfolio position",
        "Hourly aggregated long and short exposure"
    )

    st.line_chart(
        naive(res["net"]["net_mwh"]),
        use_container_width=True
    )

    st.markdown("#### Position summary")

    left, right = st.columns([1, 2])

    with left:
        st.json(
            res["position"],
            expanded=True
        )

    with right:
        st.dataframe(
            res["net"],
            use_container_width=True
        )


# ============================================================
# 04 MARK TO MARKET & P&L
# ============================================================

with tabs[3]:

    section_title(
        "Market valuation",
        "Price curves, realized DAM pricing and mark-to-market exposure"
    )

    curve = pd.DataFrame({"price": prices})

    if is_proxy is not None:

        curve["DAM actual"] = prices.where(~is_proxy)
        curve["Forward proxy"] = prices.where(is_proxy)

        curve = curve[["DAM actual", "Forward proxy"]]

    curve.index = curve.index.tz_localize(None)

    st.markdown("#### HB_HOUSTON market price curve")

    st.line_chart(
        curve,
        use_container_width=True
    )

    st.divider()

    st.markdown("#### Mark-to-market P&L")

    st.caption(
        "MtM = signed MWh × (market price − contract price), "
        "summed over delivery hours."
    )

    base = res["mtm_base"]

    daily_pnl = (
        base.groupby(
            base["delivery_start"].dt.date
        )["mtm_pnl"].sum()
    )

    st.bar_chart(
        daily_pnl,
        use_container_width=True
    )

    st.markdown("#### P&L by trade")

    st.dataframe(
        res["by_trade"],
        hide_index=True,
        use_container_width=True
    )

    with st.expander("Detailed hourly MtM breakdown"):

        st.dataframe(
            base[
                [
                    "trade_id",
                    "delivery_start",
                    "signed_mwh",
                    "trade_price",
                    "market_price",
                    "mtm_pnl"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# 05 STRESS TESTING
# ============================================================

with tabs[4]:

    section_title(
        "Market stress testing",
        "Evaluate portfolio sensitivity to market price movements"
    )

    delta = res["stress_total"] - res["base_total"]

    a, b, c = st.columns(3)

    a.metric("Base MtM", usd(res["base_total"]))

    b.metric("Stressed MtM", usd(res["stress_total"]))

    c.metric("P&L impact", usd(delta))

    st.divider()

    st.markdown("#### Scenario configuration")

    st.write(
        f"All hours: **{pct:+d}%** | "
        f"Absolute shift: **{abs_shift:+d} $/MWh** | "
        f"Evening ramp HE17-21: **{peak:+d}%**"
    )

    both = pd.DataFrame({
        "Base price": (
            res["mtm_base"]
            .groupby("delivery_start")["market_price"]
            .mean()
        ),
        "Stressed price": (
            res["mtm_stress"]
            .groupby("delivery_start")["market_price"]
            .mean()
        )
    })

    both.index = both.index.tz_localize(None)

    st.markdown("#### Base vs stressed market prices")

    st.line_chart(
        both,
        use_container_width=True
    )

    st.markdown("#### Scenario results")

    st.dataframe(
        res["scenario_table"],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("#### Scenario P&L impact")

    st.bar_chart(
        res["scenario_table"].set_index("scenario")[
            "delta_vs_base_usd"
        ],
        use_container_width=True
    )


# ============================================================
# 06 VALUE AT RISK
# ============================================================

with tabs[5]:

    section_title(
        "Value at Risk",
        "P&L risk of the hourly net position from HB_HOUSTON DAM price history"
    )

    if True:
        try:
            if source == "ERCOT API (live)":
                hist_series = get_history_live(get_client(*creds), int(var_days), str(date.today()))
            else:
                hist_series = get_history_synth(int(var_days), str(date.today()))
            hist_curve = history_to_curve(hist_series)
            if len(hist_curve) < 30:
                raise ValueError(f"Need at least 30 days of history, got {len(hist_curve)}")
            pos = position_profile(res["net"])
            out = var_summary(hist_curve, pos, conf=var_conf, horizon_days=var_horizon)
        except (ValueError, ErcotAPIError, requests.RequestException) as e:
            st.error(f"VaR input problem: {e}")
        else:
            tbl = out["table"]

            for col, (_, r) in zip(st.columns(3), tbl.iterrows()):
                col.metric(
                    f"{r['method']} VaR",
                    usd(r["var_usd"]),
                    f"ES {usd(r['es_usd'])}",
                    delta_color="off"
                )

            st.caption(
                f"{out['days']} days of DAM history · {var_conf:.0%} confidence · "
                f"{var_horizon}-day horizon · ES = average loss beyond VaR"
            )

            st.divider()

            st.markdown("#### Historical 1-day P&L distribution")
            counts = pd.cut(pd.Series(out["hist_pnl"]), 30).value_counts().sort_index()
            counts.index = [round(i.mid) for i in counts.index]
            st.bar_chart(counts, use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.markdown("#### Position used (avg MW by hour ending)")
                st.bar_chart(pos, use_container_width=True)
            with right:
                st.markdown("#### VaR by method")
                st.dataframe(tbl, hide_index=True, use_container_width=True)

# ============================================================
# 07 AI COPILOT
# ============================================================

with tabs[6]:

    section_title(
        "AI trading copilot",
        "Ask questions about portfolio exposure, MtM, pricing and stress results"
    )

    ctx = build_context(res)

    key = secret("ANTHROPIC_API_KEY")

    if not key:

        key = st.text_input(
            "Anthropic API key (optional)",
            type="password"
        ) or None

        st.caption(
            "Without an API key, the deterministic rule-based "
            "explainer is used."
        )

    st.markdown("#### Ask your trading copilot")

    q = st.text_input(
        "Question",
        placeholder="Why does the SELL trade lose money in this scenario?"
    )

    if st.button(
        "Generate AI explanation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing portfolio and generating insights..."):

            text, src = explain(
                ctx,
                question=q or None,
                api_key=key
            )

        st.markdown("### Copilot insights")

        st.markdown(text)

        st.caption(f"Explanation source: {src}")

    with st.expander("View context provided to AI"):

        st.caption(
            "The model explains calculated results; "
            "it does not calculate the underlying trading numbers."
        )

        st.json(ctx, expanded=False)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    GENAI HOUSTON HUB ETRM &nbsp; | &nbsp; ERCOT HB_HOUSTON
    <br><br>
    Commodity Analytics · Portfolio Risk · Market Intelligence
    <br><br>
    <span style="font-size:10px; color:#526a82;">
        Created by Preet Modi
    </span>
</div>
""", unsafe_allow_html=True)
