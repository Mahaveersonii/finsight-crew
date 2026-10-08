"""Market + fundamental data with a 3-level fallback chain.

  prices:        Yahoo Finance  ->  local cache (stale, flagged)
  fundamentals:  Yahoo Finance  ->  SEC EDGAR XBRL company facts  ->  local cache

Every fallback is written to the `events` table so it shows up in Grafana
and in the demo video ("error recovery").
"""
import json
import logging
import time
from datetime import datetime

import pandas as pd
import requests

from .. import config, db

log = logging.getLogger(__name__)
SEC_HEADERS = {"User-Agent": config.SEC_IDENTITY, "Accept-Encoding": "gzip, deflate"}


def _cache_path(name: str):
    return config.CACHE_DIR / name


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

_PERIOD_DAYS = {"5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 10**6}


def get_price_history(ticker: str, period: str = "2y", run_id=None) -> pd.DataFrame:
    """Daily OHLCV. Falls back to cached data if Yahoo fails.

    Each period has its own cache file: the 15-minute price check fetches only 5 days, and a
    shared file would let that overwrite the 2-year history the agents fall back on."""
    ticker = ticker.upper()
    cache = _cache_path(f"prices_{ticker}_{period}.pkl")
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is not None:
            # Before the open Yahoo can add a row for today with volume but no prices; one blank
            # last row would turn every indicator, the stop-loss and the share count into NaN.
            df = df.dropna(subset=["Close"])
        if df is None or df.empty:
            raise ValueError("empty price history")
        df.index = df.index.tz_localize(None)
        df.to_pickle(cache)
        return df
    except Exception as exc:  # noqa: BLE001
        db.log_event("data_fallback", "prices:yahoo->cache", f"{ticker} ({period}): {exc}", run_id)
        # Best cached copy: the requested period, else the longest one we have (trimmed to the period).
        want = _PERIOD_DAYS.get(period, 504)
        candidates = sorted(config.CACHE_DIR.glob(f"prices_{ticker}_*.pkl"),
                            key=lambda p: (p != cache, -len(pd.read_pickle(p))))
        for path in candidates:
            df = pd.read_pickle(path)
            if len(df) >= min(want, 60) or path == cache:
                df = df.iloc[-want:].copy()
                df.attrs["stale"] = True
                return df
        raise RuntimeError(f"No price data available for {ticker}") from exc


def latest_price(ticker: str) -> float:
    df = get_price_history(ticker, period="5d")
    return float(df["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

_YF_FIELDS = [
    "longName", "sector", "industry", "currentPrice", "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda", "ebitda",
    "totalRevenue", "revenueGrowth", "earningsGrowth", "grossMargins", "operatingMargins",
    "profitMargins", "returnOnEquity", "totalDebt", "totalCash", "debtToEquity",
    "freeCashflow", "operatingCashflow", "sharesOutstanding", "beta", "dividendYield",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "trailingEps", "targetMeanPrice",
    "recommendationKey", "currency", "financialCurrency",
]


def _from_yahoo(ticker: str) -> dict:
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    data = {k: info.get(k) for k in _YF_FIELDS}
    if not data.get("marketCap"):
        raise ValueError("Yahoo returned no fundamentals")
    data["source"] = "Yahoo Finance"
    return data


def _cik_for(ticker: str) -> str:
    cache = _cache_path("sec_tickers.json")
    if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
        mapping = json.loads(cache.read_text())
    else:
        raw = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20).json()
        mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
        cache.write_text(json.dumps(mapping))
    if ticker not in mapping:
        raise ValueError(f"{ticker} not found in SEC ticker list")
    return mapping[ticker]


ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "40-F"}
MAX_AGE_DAYS = 550  # ignore annual figures older than ~18 months


def _latest_annual(facts: dict, *concepts, namespace: str = "us-gaap"):
    """Most recent annual value across *all* candidate concepts (companies switch tags
    over time, e.g. Apple moved from `Revenues` to `RevenueFromContract...`), rejecting
    stale figures so a 2013 debt number can never masquerade as current."""
    ns = facts.get("facts", {}).get(namespace, {})
    best = None
    for concept in concepts:
        for unit_vals in ns.get(concept, {}).get("units", {}).values():
            for u in unit_vals:
                annual = u.get("form") in ANNUAL_FORMS and u.get("fp") == "FY"
                if (annual or namespace == "dei") and u.get("end") and (best is None or u["end"] > best["end"]):
                    best = u
    if not best:
        return None
    age = (datetime.now() - datetime.strptime(best["end"], "%Y-%m-%d")).days
    return best["val"] if age <= MAX_AGE_DAYS else None


def _from_sec(ticker: str) -> dict:
    cik = _cik_for(ticker)
    facts = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=SEC_HEADERS, timeout=30
    ).json()
    revenue = _latest_annual(facts, "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")
    net_income = _latest_annual(facts, "NetIncomeLoss", "ProfitLoss")
    ocf = _latest_annual(facts, "NetCashProvidedByUsedInOperatingActivities")
    capex = _latest_annual(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
    shares = (_latest_annual(facts, "EntityCommonStockSharesOutstanding", namespace="dei")
              or _latest_annual(facts, "WeightedAverageNumberOfDilutedSharesOutstanding"))
    cash = _latest_annual(facts, "CashAndCashEquivalentsAtCarryingValue",
                          "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
    debt = _latest_annual(facts, "LongTermDebt", "LongTermDebtNoncurrent")
    if not revenue or not shares:
        # Incomplete / stale filings (e.g. a newly reorganised holding company with no 10-K yet):
        # better to fall through to the cache than return half-empty numbers.
        raise ValueError(f"SEC XBRL has no current annual revenue/shares for {ticker}")
    price = latest_price(ticker)
    mcap = price * shares
    return {
        "longName": facts.get("entityName"),
        "currentPrice": price,
        "marketCap": mcap,
        "totalRevenue": revenue,
        "profitMargins": (net_income / revenue) if net_income is not None else None,
        "trailingPE": (mcap / net_income) if net_income and net_income > 0 else None,
        "operatingCashflow": ocf,
        "freeCashflow": (ocf - capex) if ocf is not None and capex is not None else None,
        "sharesOutstanding": shares,
        "totalCash": cash,
        "totalDebt": debt,
        "enterpriseValue": mcap + (debt or 0) - (cash or 0),
        "source": "SEC EDGAR XBRL (fallback)",
    }


def get_fundamentals(ticker: str, run_id=None) -> dict:
    ticker = ticker.upper()
    cache = _cache_path(f"fund_{ticker}.json")
    try:
        data = _from_yahoo(ticker)
    except Exception as exc:  # noqa: BLE001
        db.log_event("data_fallback", "fundamentals:yahoo->sec", f"{ticker}: {exc}", run_id)
        try:
            if config.M["corpus"] != "sec":
                # India has no free, official structured-data API like SEC XBRL: go straight to the cache.
                raise ValueError("no official structured-data fallback for this market")
            data = _from_sec(ticker)
        except Exception as exc2:  # noqa: BLE001
            db.log_event("data_fallback", "fundamentals:sec->cache", f"{ticker}: {exc2}", run_id)
            if cache.exists():
                data = json.loads(cache.read_text())
                data["source"] = f"{data.get('source')} (cached {datetime.fromtimestamp(cache.stat().st_mtime):%Y-%m-%d})"
                return data
            raise RuntimeError(f"No fundamentals available for {ticker}") from exc2
    cache.write_text(json.dumps(data, default=str))
    return data


def get_fcf_history(ticker: str) -> list:
    """Annual free cash flow, oldest first (used by the DCF)."""
    try:
        import yfinance as yf

        cf = yf.Ticker(ticker).cashflow
        if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
            return []
        series = cf.loc["Free Cash Flow"].dropna().sort_index()
        return [float(v) for v in series.values]
    except Exception:  # noqa: BLE001
        return []


def get_news(ticker: str, limit: int = 6) -> list:
    try:
        import yfinance as yf

        items = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in items[:limit]:
        c = it.get("content", it)
        out.append({
            "title": c.get("title"),
            "publisher": (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher"),
            "date": c.get("pubDate") or c.get("providerPublishTime"),
        })
    return out
