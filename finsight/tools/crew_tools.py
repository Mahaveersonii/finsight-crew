"""CrewAI tool wrappers. Each tool:
  - takes a plain ticker / question string (easy for an 8B model to call)
  - returns compact JSON text (keeps the context window small)
  - is timed and logged to the `events` table (tool-call audit trail)
  - never raises: errors come back as {"error": ...} so the agent can recover
"""
import functools
import json
import time

from crewai.tools import tool

from .. import broker, db, rag
from . import market_data as md
from . import valuation as val

# Per-run context: run id for logging + a cache so each API is hit once per run.
RUN = {"run_id": None, "cache": {}, "called": [], "on_event": None}


def reset_run(run_id):
    RUN["run_id"] = run_id
    RUN["cache"] = {}
    RUN["called"] = []


def _cached(key, fn):
    if key not in RUN["cache"]:
        RUN["cache"][key] = fn()
    return RUN["cache"][key]


def _logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        RUN["called"].append(fn.__name__)
        if RUN["on_event"]:
            RUN["on_event"](f"🔧 {fn.__name__}({', '.join(map(str, [*args, *kwargs.values()]))})")
        try:
            out = fn(*args, **kwargs)
            db.log_event("tool_call", fn.__name__, json.dumps({"args": args, "kwargs": kwargs})[:500],
                         RUN["run_id"], (time.time() - t0) * 1000)
            return out
        except Exception as exc:  # noqa: BLE001
            db.log_event("error", fn.__name__, str(exc), RUN["run_id"], (time.time() - t0) * 1000)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "hint": "Continue with the data you have and say what is missing."})
    return wrapper


def _clean_ticker(t: str) -> str:
    return str(t).strip().strip("'\"").upper().split()[0]


# --- shared data accessors (also used by the pipeline) ----------------------

def prices(ticker):
    return _cached(("px", ticker), lambda: md.get_price_history(ticker, run_id=RUN["run_id"]))


def fundamentals(ticker):
    return _cached(("fund", ticker), lambda: md.get_fundamentals(ticker, run_id=RUN["run_id"]))


def tech(ticker):
    return _cached(("tech", ticker), lambda: val.technicals(prices(ticker)))


def valuation(ticker):
    return _cached(("val", ticker), lambda: val.valuation_report(fundamentals(ticker), md.get_fcf_history(ticker), tech(ticker)))


# --- Data Extractor tools ---------------------------------------------------

@tool("get_market_snapshot")
@_logged
def get_market_snapshot(ticker: str) -> str:
    """Live price and technical indicators for a US stock ticker (e.g. 'AAPL'):
    last close, 50/200-day moving averages, trend, RSI, ATR, 1-12 month returns,
    volatility and 1-year max drawdown."""
    t = _clean_ticker(ticker)
    f = fundamentals(t)
    snap = {"ticker": t, "company": f.get("longName"), "sector": f.get("sector"), "industry": f.get("industry"),
            "market_cap_bn": round(f["marketCap"] / 1e9, 1) if f.get("marketCap") else None,
            "52w_high": f.get("fiftyTwoWeekHigh"), "52w_low": f.get("fiftyTwoWeekLow"), **tech(t)}
    return json.dumps(snap)


@tool("get_fundamentals")
@_logged
def get_fundamentals(ticker: str) -> str:
    """Company fundamentals for a US stock ticker: revenue, margins, growth, cash,
    debt, free cash flow, P/E, EV/EBITDA, beta and analyst target. Also reports
    the data source used (Yahoo Finance, or SEC EDGAR as fallback)."""
    t = _clean_ticker(ticker)
    f = fundamentals(t)
    keep = ["longName", "currentPrice", "marketCap", "enterpriseValue", "totalRevenue", "revenueGrowth",
            "earningsGrowth", "grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", "freeCashflow",
            "totalCash", "totalDebt", "debtToEquity", "trailingPE", "forwardPE", "enterpriseToEbitda", "beta",
            "dividendYield", "targetMeanPrice", "recommendationKey", "source"]
    out = {}
    for k in keep:
        v = f.get(k)
        if isinstance(v, (int, float)) and abs(v) >= 1e6:
            v = f"{v / 1e9:.2f}bn"
        elif isinstance(v, float):
            v = round(v, 4)
        out[k] = v
    return json.dumps(out)


# --- Financial Analyst tools ------------------------------------------------

@tool("run_valuation")
@_logged
def run_valuation(ticker: str) -> str:
    """Valuation model for a ticker: P/E, EV/EBITDA, FCF yield, ROE, leverage,
    a 3-scenario (bear/base/bull) discounted-cash-flow intrinsic value per share,
    margin of safety versus the current price, and automatic risk flags."""
    t = _clean_ticker(ticker)
    return json.dumps(valuation(t))


@tool("search_sec_filings")
@_logged
def search_sec_filings(ticker: str, question: str) -> str:
    """Search the company's latest SEC 10-K annual report (Business, Risk Factors,
    MD&A sections) and return the most relevant passages with citations.
    Use it for qualitative evidence: risks, competition, growth drivers, strategy,
    regulation, margins. Always quote the citation tag you use."""
    t = _clean_ticker(ticker)
    if t not in rag.indexed_tickers():
        rag.ingest_ticker(t)
    hits = rag.search(t, question, k=3, run_id=RUN["run_id"])
    return json.dumps([{"citation": h["citation"], "relevance": h["score"], "text": h["text"][:700]} for h in hits])


@tool("get_past_decisions")
@_logged
def get_past_decisions(ticker: str) -> str:
    """Agent memory: the fund's previous signals for this ticker (date, action,
    confidence, price, outcome) and whether we currently hold it. Use it to stay
    consistent with, or explicitly explain changes from, earlier decisions."""
    t = _clean_ticker(ticker)
    past = [{"date": str(s["ts"])[:16], "action": s["action"], "confidence": s["confidence"], "price": s["price"],
             "status": s["status"], "rationale": (s["rationale"] or "")[:200]} for s in db.recent_signals(t, 3)]
    held = next((p for p in broker.open_positions() if p["ticker"] == t), None)
    return json.dumps({"previous_signals": past or "none - first time analysing this ticker",
                       "current_position": held and {k: held[k] for k in ("shares", "avg_price", "stop_loss", "take_profit")}},
                      default=str)


# --- Portfolio Manager tools ------------------------------------------------

@tool("get_portfolio_state")
@_logged
def get_portfolio_state() -> str:
    """Current paper portfolio: cash, equity, open positions, sector weights and
    the fund's hard risk limits. Takes no input."""
    return json.dumps(broker.portfolio_summary())


@tool("plan_position")
@_logged
def plan_position(ticker: str) -> str:
    """Risk-managed trade plan for a prospective BUY of `ticker`: ATR-based
    stop-loss, 2R take-profit, share count from the 1%-risk rule, capped by
    position, sector and cash limits. Returns the binding constraint."""
    t = _clean_ticker(ticker)
    f = fundamentals(t)
    tc = tech(t)
    return json.dumps(broker.plan_trade(t, tc["last_close"], tc["atr_14"], f.get("sector") or "Unknown"))


EXTRACTOR_TOOLS = [get_market_snapshot, get_fundamentals]
ANALYST_TOOLS = [run_valuation, search_sec_filings, get_past_decisions]
PM_TOOLS = [get_portfolio_state, plan_position]
