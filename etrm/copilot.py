"""AI copilot: turns pipeline results into a grounded explanation.

Design: the LLM never computes numbers. We hand it a compact JSON context built from
the engine's outputs and instruct it to use only those figures. If no API key is set
or the call fails, a deterministic rule-based explainer produces the same structure.
"""
from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a risk analyst copilot embedded in an energy trading (ETRM) tool.
You explain portfolio position, mark-to-market (MtM) P&L and price-stress results to a
trader or risk manager.
Rules:
- Use ONLY the numbers in the JSON context. Never invent or recompute figures.
- Sign convention: positive net MWh = long (we hold more power than we owe); negative = short.
- BUY gains when market > contract price; SELL gains when market < contract price.
- Be concise: a 2-sentence headline, 3-5 bullets on drivers (position shape, which trade
  drives P&L, which hours hurt/help), and one suggested hedge or follow-up.
- Flag assumptions: say how many hours use published DAM prices vs the forward proxy (price_data),
  that the proxy is not a real forward curve, and that on-peak ignores NERC holidays."""


def usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _round(x, n=0):
    return round(float(x), n)


def build_context(res: dict) -> dict:
    base, stressed = res["mtm_base"], res["mtm_stress"]
    sc = res["stress"]
    worst = stressed.nsmallest(3, "pnl_delta")
    best = stressed.nlargest(3, "pnl_delta")

    def hour_rows(df):
        return [
            {
                "trade_id": r.trade_id,
                "delivery_start": str(r.delivery_start),
                "market_price_base": _round(base.loc[i, "market_price"], 2),
                "market_price_stressed": _round(r.market_price, 2),
                "pnl_delta_usd": _round(r.pnl_delta),
            }
            for i, r in df.iterrows()
        ]

    return {
        "trades": [t.to_dict() for t in res["trades"]],
        "position": {k: _round(v, 1) for k, v in res["position"].items()},
        "base_mtm_usd": _round(res["base_total"]),
        "stress_scenario": {
            "name": sc.name,
            "pct_shift": sc.pct_shift,
            "abs_shift_usd_mwh": sc.abs_shift,
            "evening_peak_pct_shift_HE17_21": sc.peak_pct_shift,
        },
        "stress_mtm_usd": _round(res["stress_total"]),
        "stress_delta_usd": _round(res["stress_total"] - res["base_total"]),
        "by_trade": [
            {k: (_round(v) if k != "trade_id" else v) for k, v in row.items()}
            for row in res["by_trade"].to_dict("records")
        ],
        "worst_hours_under_stress": hour_rows(worst),
        "best_hours_under_stress": hour_rows(best),
        "price_data": res.get("price_meta", {}),
        "avg_market_price_base": _round(base["market_price"].mean(), 2),
        "avg_market_price_stressed": _round(stressed["market_price"].mean(), 2),
    }


def rule_based_explanation(ctx: dict) -> str:
    pos, d = ctx["position"], ctx["stress_delta_usd"]
    pd_ = ctx.get("price_data", {})
    lead = max(ctx["by_trade"], key=lambda r: abs(r["delta"]))
    direction = "gains" if d >= 0 else "loses"
    book = "net long" if pos["net_mwh_total"] > 0 else "net short" if pos["net_mwh_total"] < 0 else "flat"
    sc = ctx["stress_scenario"]
    lines = [
        f"**Headline.** The book is {book} ({pos['net_mwh_total']:,.0f} MWh over {pos['hours']:,.0f} hours) with base MtM of "
        f"{usd(ctx['base_mtm_usd'])}. Under '{sc['name']}' the portfolio {direction} {usd(abs(d))}, "
        f"moving MtM to {usd(ctx['stress_mtm_usd'])}.",
        "",
        f"- **Position shape:** net position ranges from {pos['min_net_mw']:,.0f} to {pos['max_long_mw']:,.0f} MW; "
        f"{pos['hours_long']:,.0f} hours long, {pos['hours_short']:,.0f} short, {pos['hours_flat']:,.0f} flat.",
        f"- **Main driver:** trade {lead['trade_id']} moves by {usd(lead['delta'])} "
        f"(base {usd(lead['base_mtm'])} to {usd(lead['stress_mtm'])}).",
        f"- **Prices:** average market price goes from ${ctx['avg_market_price_base']:,.2f} to "
        f"${ctx['avg_market_price_stressed']:,.2f}/MWh.",
    ]
    if ctx["worst_hours_under_stress"]:
        w = ctx["worst_hours_under_stress"][0]
        lines.append(f"- **Worst hour:** {w['delivery_start']} on {w['trade_id']}, {usd(w['pnl_delta_usd'])} vs base.")
    lines += [
        f"- **Price data:** {pd_.get('source', 'unspecified')}"
        + (f"; {pd_['hours_dam_actual']:,} hours use published DAM prices and {pd_['hours_forward_proxy']:,} hours use a forward proxy ({pd_['proxy_method']})."
           if pd_.get("hours_total") else "."),
        "",
        "**Follow-up.** If the stressed move is larger than your limit, consider hedging the open "
        "net position in the hours that drive the loss (e.g. an on-peak or evening block).",
        "",
        "_Assumptions: forward-proxy hours are not a market forward curve; on-peak definition ignores NERC holidays._",
    ]
    return "\n".join(lines)


def explain(ctx: dict, question: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """Return (text, source). source is 'claude:<model>' or 'rule-based'."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return rule_based_explanation(ctx), "rule-based (no API key set)"
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        user = (
            f"Context (JSON):\n{json.dumps(ctx, indent=2)}\n\n"
            + (f"Question: {question}" if question else "Explain these results.")
        )
        msg = client.messages.create(
            model=model, max_tokens=800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, f"claude:{model}"
    except Exception as e:  # network/auth/model errors -> graceful fallback
        return rule_based_explanation(ctx), f"rule-based (LLM call failed: {type(e).__name__})"
