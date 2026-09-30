"""Deterministic finance maths. The LLM never does arithmetic itself -
it calls these functions through tools and reasons about the results."""
import math

import numpy as np
import pandas as pd

from .. import config

# Market assumptions (see markets.py): US = 10y Treasury 4.3%, ERP 5.5%, terminal 2.5%;
# India = 10y G-sec 6.5%, ERP 7%, terminal 5%.
UNIT, UNIT_SIZE = config.M["big_unit"]   # "bn" (1e9) for the US, "crore" (1e7) for India
RISK_FREE = config.M["risk_free"]
EQUITY_RISK_PREMIUM = config.M["equity_risk_premium"]
TERMINAL_GROWTH = config.M["terminal_growth"]
COE_LOW, COE_HIGH = config.M["cost_of_equity_band"]


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), n)


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, n: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def technicals(df: pd.DataFrame) -> dict:
    close = df["Close"]
    last = float(close.iloc[-1])

    def ret(days):
        return _r((last / float(close.iloc[-days - 1]) - 1) * 100, 1) if len(close) > days else None

    one_year = close.iloc[-252:]
    drawdown = (one_year / one_year.cummax() - 1).min()
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    trend = "uptrend" if sma200 and last > sma50 > sma200 else "downtrend" if sma200 and last < sma50 < sma200 else "mixed"
    return {
        "last_close": _r(last),
        "as_of": str(close.index[-1].date()),
        "sma_50": _r(sma50),
        "sma_200": _r(sma200),
        "trend": trend,
        "rsi_14": _r(rsi(close), 1),
        "atr_14": _r(atr(df)),
        "return_1m_pct": ret(21),
        "return_3m_pct": ret(63),
        "return_6m_pct": ret(126),
        "return_12m_pct": ret(252),
        "volatility_annual_pct": _r(close.pct_change().iloc[-252:].std() * math.sqrt(252) * 100, 1),
        "max_drawdown_1y_pct": _r(drawdown * 100, 1),
        "stale_data": bool(df.attrs.get("stale", False)),
    }


def beta_vs(prices: pd.DataFrame, bench: pd.DataFrame):
    """Beta against the market's own index from 2 years of weekly returns (Yahoo's beta for Indian
    stocks is measured against the S&P 500 and can come out near zero or negative)."""
    a = prices["Close"].resample("W-FRI").last().pct_change()
    b = bench["Close"].resample("W-FRI").last().pct_change()
    df = pd.concat([a, b], axis=1, keys=["s", "m"]).dropna().iloc[-104:]
    if len(df) < 52 or df["m"].var() == 0:
        return None
    return round(float(df["s"].cov(df["m"]) / df["m"].var()), 2)


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

def cost_of_equity(beta) -> float:
    beta = beta if beta and 0.3 < beta < 3 else 1.0
    return min(max(RISK_FREE + beta * EQUITY_RISK_PREMIUM, COE_LOW), COE_HIGH)


def dcf(fcf_base: float, growth: float, discount: float, shares: float,
        cash: float = 0.0, debt: float = 0.0, years: int = 10) -> dict:
    """Two-stage FCF-to-firm DCF: `years` of explicit growth fading linearly to
    the terminal rate, then a Gordon-growth terminal value."""
    flows, pv = [], 0.0
    fcf = fcf_base
    for year in range(1, years + 1):
        g = growth + (TERMINAL_GROWTH - growth) * (year - 1) / years
        fcf *= 1 + g
        disc = fcf / (1 + discount) ** year
        flows.append(round(fcf / UNIT_SIZE, 2))
        pv += disc
    terminal = fcf * (1 + TERMINAL_GROWTH) / (discount - TERMINAL_GROWTH)
    pv_terminal = terminal / (1 + discount) ** years
    enterprise = pv + pv_terminal
    equity = enterprise + (cash or 0) - (debt or 0)
    return {
        "intrinsic_value_per_share": _r(equity / shares) if shares else None,
        f"enterprise_value_{UNIT}": _r(enterprise / UNIT_SIZE, 1),
        "terminal_share_of_value_pct": _r(pv_terminal / enterprise * 100, 1),
        f"projected_fcf_{UNIT}": flows,
    }


def implied_growth(price, fcf_base, discount, shares, cash, debt):
    """Reverse DCF: the FCF growth rate (years 1-10) the current price implies.
    Bisection on the monotonic DCF value."""
    lo, hi = -0.10, 0.60
    for _ in range(60):
        mid = (lo + hi) / 2
        iv = dcf(fcf_base, mid, discount, shares, cash, debt)["intrinsic_value_per_share"] or 0
        lo, hi = (mid, hi) if iv < price else (lo, mid)
    return round(mid * 100, 1) if lo > -0.099 and hi < 0.599 else None


def _clamp(x, a=0.0, b=100.0):
    return max(a, min(b, x))


def quant_score(r: dict, tech: dict, mos, upside) -> dict:
    """Transparent 0-100 factor scores (value / quality / momentum)."""
    value = 0.4 * _clamp(50 + (mos or -50) / 2) + 0.3 * _clamp((r["fcf_yield_pct"] or 0) / 5 * 100) + \
        0.3 * _clamp(50 + (upside or 0) * 2)
    de = r["debt_to_equity"]
    quality = 0.4 * _clamp((r["roe_pct"] or 0) / 25 * 100) + 0.3 * _clamp((r["net_margin_pct"] or 0) / 20 * 100) + \
        0.3 * (100 if de is None or de < 100 else 50 if de < 200 else 0)
    trend = {"uptrend": 100, "mixed": 50, "downtrend": 0}[tech.get("trend", "mixed")]
    rsi_ok = 100 if 40 <= (tech.get("rsi_14") or 50) <= 70 else 40
    momentum = 0.5 * _clamp(50 + (tech.get("return_12m_pct") or 0)) + 0.3 * trend + 0.2 * rsi_ok
    composite = 0.35 * value + 0.35 * quality + 0.30 * momentum
    return {"value": round(value), "quality": round(quality), "momentum": round(momentum), "composite": round(composite),
            "method": "value 35% (DCF margin of safety, FCF yield, analyst upside) + quality 35% (ROE, net margin, leverage) + momentum 30% (12m return, trend, RSI)"}


def valuation_report(fund: dict, fcf_history: list, tech: dict) -> dict:
    price = fund.get("currentPrice") or tech.get("last_close")
    shares = fund.get("sharesOutstanding")
    # Prefer reported FCF (operating cash flow - capex) from the latest annual cash-flow
    # statement; Yahoo's `freeCashflow` field is a levered estimate that can be far lower.
    fcf_now = fcf_history[-1] if fcf_history else fund.get("freeCashflow")

    # Growth assumption: blend of revenue growth and FCF CAGR, clamped to a sane band.
    candidates = []
    if fund.get("revenueGrowth") is not None:
        candidates.append(fund["revenueGrowth"])
    if len(fcf_history) >= 3 and fcf_history[0] > 0 and fcf_history[-1] > 0:
        candidates.append((fcf_history[-1] / fcf_history[0]) ** (1 / (len(fcf_history) - 1)) - 1)
    growth = min(max(float(np.median(candidates)) if candidates else 0.05, -0.05), 0.25)
    wacc = cost_of_equity(fund.get("beta"))

    out = {
        "price": _r(price),
        "ratios": {
            "pe_trailing": _r(fund.get("trailingPE"), 1),
            "pe_forward": _r(fund.get("forwardPE"), 1),
            "ev_to_ebitda": _r(fund.get("enterpriseToEbitda") or (
                fund["enterpriseValue"] / fund["ebitda"] if fund.get("enterpriseValue") and fund.get("ebitda") else None), 1),
            "price_to_book": _r(fund.get("priceToBook"), 1),
            "fcf_yield_pct": _r(fcf_now / fund["marketCap"] * 100, 2)
            if fcf_now and fund.get("marketCap") and fund.get("sector") != "Financial Services" else None,
            "net_margin_pct": _r((fund.get("profitMargins") or 0) * 100, 1) if fund.get("profitMargins") is not None else None,
            "roe_pct": _r((fund.get("returnOnEquity") or 0) * 100, 1) if fund.get("returnOnEquity") is not None else None,
            "debt_to_equity": _r(fund.get("debtToEquity"), 1),
            "revenue_growth_pct": _r((fund.get("revenueGrowth") or 0) * 100, 1) if fund.get("revenueGrowth") is not None else None,
        },
        "dcf": None,
        "assumptions": {"growth_pct": _r(growth * 100, 1), "discount_rate_pct": _r(wacc * 100, 1),
                        "terminal_growth_pct": TERMINAL_GROWTH * 100},
    }

    currency_mismatch = (fund.get("financialCurrency") and fund.get("currency")
                         and fund["financialCurrency"] != fund["currency"])
    if currency_mismatch:
        out["dcf"] = {"note": f"DCF not meaningful: statements are in {fund['financialCurrency']} but the share "
                              f"trades in {fund['currency']}."}
    elif fcf_now and fcf_now > 0 and shares:
        cash, debt = fund.get("totalCash") or 0, fund.get("totalDebt") or 0
        scenarios = {
            "bear": dcf(fcf_now, growth - 0.04, wacc + 0.01, shares, cash, debt),
            "base": dcf(fcf_now, growth, wacc, shares, cash, debt),
            "bull": dcf(fcf_now, growth + 0.03, wacc - 0.01, shares, cash, debt),
        }
        base_iv = scenarios["base"]["intrinsic_value_per_share"]
        out["dcf"] = {
            "scenarios": {k: v["intrinsic_value_per_share"] for k, v in scenarios.items()},
            "base_case_detail": scenarios["base"],
            "margin_of_safety_pct": _r((base_iv - price) / price * 100, 1) if base_iv and price else None,
            "market_implied_fcf_growth_pct": implied_growth(price, fcf_now, wacc, shares, cash, debt),
            "note": "10-year two-stage DCF; growth fades linearly to terminal rate. market_implied_fcf_growth_pct is a reverse DCF: the growth the current price requires.",
        }
    else:
        out["dcf"] = {"note": "DCF not meaningful: free cash flow is negative or unavailable (typical for banks, "
                            "whose cash flows include deposits and loans). Rely on P/E, P/B and ROE instead."}

    # Simple, transparent risk flags the analyst must address.
    flags = []
    r = out["ratios"]
    if r["pe_trailing"] and r["pe_trailing"] > 40:
        flags.append(f"Rich valuation: trailing P/E {r['pe_trailing']}")
    if r["debt_to_equity"] and r["debt_to_equity"] > 150:
        flags.append(f"High leverage: debt/equity {r['debt_to_equity']}%")
    if tech.get("volatility_annual_pct") and tech["volatility_annual_pct"] > 45:
        flags.append(f"High volatility: {tech['volatility_annual_pct']}% annualised")
    if tech.get("max_drawdown_1y_pct") and tech["max_drawdown_1y_pct"] < -30:
        flags.append(f"Deep 1y drawdown: {tech['max_drawdown_1y_pct']}%")
    if tech.get("rsi_14") and tech["rsi_14"] > 75:
        flags.append(f"Overbought: RSI {tech['rsi_14']}")
    if tech.get("trend") == "downtrend":
        flags.append("Price below 50- and 200-day averages (downtrend)")
    if out["dcf"].get("margin_of_safety_pct") is not None and out["dcf"]["margin_of_safety_pct"] < -30:
        flags.append(f"Price {abs(out['dcf']['margin_of_safety_pct'])}% above base-case DCF value")
    out["risk_flags"] = flags
    target = fund.get("targetMeanPrice")
    upside = _r((target - price) / price * 100, 1) if target and price else None
    out["analyst_consensus"] = {"target_mean_price": target, "upside_pct": upside, "rating": fund.get("recommendationKey")}
    out["quant_score"] = quant_score(r, tech, (out["dcf"] or {}).get("margin_of_safety_pct"), upside)
    return out


def full_report(fund: dict, fcf_history: list, prices: pd.DataFrame, bench: pd.DataFrame = None) -> dict:
    """The single valuation path used by both the agents' tool and the Streamlit header, so the
    numbers on screen always match what the crew saw. Beta is measured against the market's own index."""
    tech = technicals(prices)
    f = dict(fund)
    b = beta_vs(prices, bench) if bench is not None and len(bench) else None
    if b is not None:
        f["beta"] = b
    rep = valuation_report(f, fcf_history, tech)
    rep["assumptions"]["beta"] = f.get("beta")
    rep["assumptions"]["beta_vs"] = config.M["benchmark_name"] if b is not None else "data vendor"
    return rep
