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

def get_price_history(ticker: str, period: str = "2y", run_id=None) -> pd.DataFrame:
    """Daily OHLCV. Falls back to the last cached copy if Yahoo fails."""
    ticker = ticker.upper()
    cache = _cache_path(f"prices_{ticker}.pkl")
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("empty price history")
        df.index = df.index.tz_localize(None)
        df.to_pickle(cache)
        return df
    except Exception as exc:  # noqa: BLE001
        db.log_event("data_fallback", "prices:yahoo->cache", f"{ticker}: {exc}", run_id)
        if cache.exists():
            df = pd.read_pickle(cache)
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
    "recommendationKey",
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


def _latest_annual(facts: dict, *concepts):
    """Most recent 10-K value among a list of candidate us-gaap concepts."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        units = gaap.get(concept, {}).get("units", {})
        for unit_vals in units.values():
            annual = [u for u in unit_vals if u.get("form") == "10-K" and u.get("fp") == "FY"]
            if annual:
                annual.sort(key=lambda u: u.get("end", ""))
                return annual[-1]["val"]
    return None


def _from_sec(ticker: str) -> dict:
    cik = _cik_for(ticker)
    facts = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=SEC_HEADERS, timeout=30
    ).json()
    revenue = _latest_annual(facts, "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
    net_income = _latest_annual(facts, "NetIncomeLoss")
    ocf = _latest_annual(facts, "NetCashProvidedByUsedInOperatingActivities")
    capex = _latest_annual(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
    shares = _latest_annual(facts, "WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding")
    cash = _latest_annual(facts, "CashAndCashEquivalentsAtCarryingValue")
    debt = _latest_annual(facts, "LongTermDebt", "LongTermDebtNoncurrent")
    price = latest_price(ticker)
    mcap = price * shares if shares else None
    return {
        "longName": facts.get("entityName"),
        "currentPrice": price,
        "marketCap": mcap,
        "totalRevenue": revenue,
        "profitMargins": (net_income / revenue) if revenue and net_income else None,
        "trailingPE": (mcap / net_income) if mcap and net_income and net_income > 0 else None,
        "operatingCashflow": ocf,
        "freeCashflow": (ocf - capex) if ocf and capex else ocf,
        "sharesOutstanding": shares,
        "totalCash": cash,
        "totalDebt": debt,
        "enterpriseValue": (mcap + (debt or 0) - (cash or 0)) if mcap else None,
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
