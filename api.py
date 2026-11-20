"""Optional FastAPI wrapper: `uvicorn api:app --reload` then open /docs."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from etrm.copilot import build_context, explain
from etrm.market_data import synthetic_hub_prices
from etrm.pipeline import run_pipeline
from etrm.pnl import Scenario
from etrm.trade import Trade

app = FastAPI(title="Houston Hub ETRM API")


class TradeIn(BaseModel):
    trade_id: str = "BUY-001"
    counterparty: str = "Counterparty A"
    side: str = "BUY"
    profile: str = "ATC"
    mw: float = Field(50, gt=0)
    price: float = 38.0
    start: str = "2026-12-01"
    end: str = "2027-01-01"


class StressIn(BaseModel):
    name: str = "API stress"
    pct_shift: float = 0
    abs_shift: float = 0
    peak_pct_shift: float = 0


class RunIn(BaseModel):
    trades: list[TradeIn]
    stress: StressIn = StressIn()
    explain: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(req: RunIn):
    try:
        trades = [Trade(**t.model_dump()) for t in req.trades]
        for t in trades:
            t.validate()
        start = min(t.start for t in trades)
        end = max(t.end for t in trades)
        res = run_pipeline(trades, synthetic_hub_prices(start, end), Scenario(**req.stress.model_dump()))
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = {
        "position": res["position"],
        "base_mtm_usd": res["base_total"],
        "stress_mtm_usd": res["stress_total"],
        "by_trade": res["by_trade"].to_dict("records"),
        "scenarios": res["scenario_table"].to_dict("records"),
    }
    if req.explain:
        out["explanation"], out["explanation_source"] = explain(build_context(res))
    return out
