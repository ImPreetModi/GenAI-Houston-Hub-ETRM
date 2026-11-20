# ⚡ Houston Hub Power Trading Copilot

A mini ETRM project on **ERCOT HB_HOUSTON** prices, pulled **live from the ERCOT Public API**:

1. Capture a **physical power purchase** (BUY, ATC / on-peak / off-peak, MW, $/MWh, delivery window)
2. Generate **hourly delivery records** (DST-aware: 23/25-hour days, hour-ending convention)
3. Compute the **net portfolio position** across trades (long/short by hour)
4. Compute **mark-to-market P&L** = signed MWh × (market − contract price)
5. Run a **price stress scenario** (all-hours %, $/MWh adder, evening-ramp shock) plus presets
6. Ask an **AI copilot** to explain the result (Claude, grounded on engine output, offline fallback)
7. Show it all in **Streamlit** (optional FastAPI wrapper in `api.py`, which uses synthetic prices)

## How prices work (read this, it's the key modelling choice)

| Delivery hours | Price used |
|---|---|
| Already published (past days, plus tomorrow once posted) | **Actual HB_HOUSTON Day-Ahead SPP** from ERCOT NP4-190-CD |
| Future hours (e.g. next month) | **Forward proxy**: average weekday/weekend × hour shape of the last N days of real DAM prices, optionally rescaled to a forward ATC level you type in |

The proxy is **not** a market forward curve. The app labels every hour's source, and the copilot is told
to say so. Replacing it with a real forward curve (broker/ICE data) is the obvious next upgrade.

## Setup: ERCOT API access (free)

1. Sign in at [ERCOT API Explorer](https://apiexplorer.ercot.com/) → **Products** → subscribe to the Public API product.
2. **Profile** → *Show* → copy the **Primary key** (your subscription key).
3. Set (env vars or `.streamlit/secrets.toml`, see `secrets.toml.example`):
   `ERCOT_USERNAME`, `ERCOT_PASSWORD`, `ERCOT_SUBSCRIPTION_KEY`.
   The username/password are the API Explorer login. Auth is documented
   [here](https://developer.ercot.com/applications/pubapi/user-guide/registration-and-authentication/):
   every call carries an ID token (valid 1 hour, minted from your login) **and** the subscription key.
4. **Run the smoke test first:** `python -m etrm.ercot_api`
   It checks the token, the product list and a HB_HOUSTON pull, and prints real rows.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_engine.py && python tests/test_ercot.py   # 19 tests, no network/credentials needed
python -m etrm.ercot_api                                    # live smoke test (needs your ERCOT creds)
streamlit run app.py
```

## Architecture

```
app.py (Streamlit)   api.py (FastAPI, optional)
        \                /
         etrm/pipeline.py ─► trade.py → portfolio.py → pnl.py
              │
   etrm/price_curve.py (DAM actual + forward proxy)
              │
   etrm/ercot_api.py  (token → paged GET → parse, DST-aware)
              │
   etrm/copilot.py (JSON context ► Claude | rule-based fallback)
```

The LLM never does arithmetic: the engine computes every number; the copilot only sees a JSON summary.

## Testing note

The ERCOT client is unit-tested against a **fake ERCOT server** (token flow, paging, 401 refresh, 429 retry,
DST fall-back day, filter fallback, no password leakage). It was written from ERCOT's documentation and an
open-source client, **not run against the live API**. Column and parameter names (`deliveryDateFrom/To`,
`settlementPoint`, `DSTFlag`) are the most likely place to need a tweak, so run the smoke test and check the
API Explorer's definition of `np4-190-cd/dam_stlmnt_pnt_prices` if anything differs.

## Deploy

**Credentials warning:** a public app holding *your* ERCOT login and Anthropic key lets any visitor trigger
API calls (and spend) on your accounts. Pick one:
- **Best for a portfolio:** deploy publicly **without** secrets → visitors get the synthetic-demo mode; show
  the live ERCOT mode via local run / screen recording / GitHub README GIF.
- **Live public demo:** put the app behind login (Streamlit Cloud private app, or Cloud Run with IAM) and set
  a spend limit on the Anthropic key. Results are cached for 1 hour, which limits ERCOT calls.

### A) Streamlit Community Cloud (free)
Push to GitHub → share.streamlit.io
