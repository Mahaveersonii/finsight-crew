"""Simple, honest backtest of the rule-based layer + the fund's risk engine.

What is tested
  Entry (checked weekly): close > SMA50 > SMA200 and 40 <= RSI(14) <= 70  (the momentum pillar of the quant score)
  Exit  (checked daily):  low <= stop  (stop = entry - 2 x ATR14)
                          high >= take-profit (2R)
                          close < SMA200 (trend broken)
  Sizing: same rules as the live broker - 1% equity risk per trade, <= 10% per position, cash-limited.

What is NOT tested
  The LLM agents and the fundamental/DCF pillars: we only have *today's* fundamentals and 10-K,
  so using them historically would be look-ahead bias. The backtest therefore validates the
  risk engine and the momentum rules, not the agents' judgement.
"""
import math

import numpy as np
import pandas as pd

from . import config
from .tools import market_data as md


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Open", "High", "Low", "Close"]].copy()
    out["sma50"] = out["Close"].rolling(50).mean()
    out["sma200"] = out["Close"].rolling(200).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([out["High"] - out["Low"], (out["High"] - out["Close"].shift()).abs(),
                    (out["Low"] - out["Close"].shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out


def run(tickers=None, period: str = "5y", start_cash: float = None) -> dict:
    tickers = tickers or config.WATCHLIST
    start_cash = start_cash or config.STARTING_CASH
    data = {t: _indicators(md.get_price_history(t, period=period)) for t in tickers}
    spy = md.get_price_history(config.M["benchmark"], period=period)["Close"]

    # Start once every ticker has a valid SMA200.
    start = max(d["sma200"].first_valid_index() for d in data.values())
    dates = sorted(set().union(*[d.loc[start:].index for d in data.values()]))

    cash, pos, trades, curve = start_cash, {}, [], []
    for i, day in enumerate(dates):
        # --- exits (daily) ---
        for t in list(pos):
            d = data[t]
            if day not in d.index:
                continue
            row, p = d.loc[day], pos[t]
            px, reason = None, None
            if row["Low"] <= p["stop"]:
                px, reason = min(p["stop"], row["Open"]), "stop_loss"
            elif row["High"] >= p["tp"]:
                px, reason = max(p["tp"], row["Open"]), "take_profit"
            elif row["Close"] < row["sma200"]:
                px, reason = row["Close"], "trend_break"
            if px is not None:
                cash += p["shares"] * px
                trades.append({"ticker": t, "entry_date": p["date"], "exit_date": day, "entry": round(p["entry"], 2),
                               "exit": round(px, 2), "shares": p["shares"], "reason": reason,
                               "pnl": round((px - p["entry"]) * p["shares"], 2),
                               "return_pct": round((px / p["entry"] - 1) * 100, 2)})
                del pos[t]

        equity = cash + sum(p["shares"] * data[t].loc[:day, "Close"].iloc[-1] for t, p in pos.items())

        # --- entries (weekly) ---
        if i % 5 == 0:
            for t, d in data.items():
                if t in pos or day not in d.index:
                    continue
                row = d.loc[day]
                if not (row["Close"] > row["sma50"] > row["sma200"] and 40 <= row["rsi"] <= 70):
                    continue
                price, stop = row["Close"], row["Close"] - config.ATR_STOP_MULT * row["atr"]
                risk = max(price - stop, price * 0.01)
                shares = min(math.floor(equity * config.RISK_PER_TRADE / risk),
                             math.floor(equity * config.MAX_POSITION_PCT / price),
                             math.floor(cash / price))
                if shares > 0:
                    cash -= shares * price
                    pos[t] = {"shares": shares, "entry": price, "stop": stop, "tp": price + 2 * (price - stop), "date": day}

        equity = cash + sum(p["shares"] * data[t].loc[:day, "Close"].iloc[-1] for t, p in pos.items())
        curve.append({"date": day, "equity": equity, "invested": 1 - cash / equity})

    curve_df = pd.DataFrame(curve).set_index("date")
    eq = curve_df["equity"]
    bench = spy.reindex(eq.index, method="ffill")
    ew = pd.concat([data[t]["Close"].reindex(eq.index, method="ffill") / data[t]["Close"].reindex(eq.index, method="ffill").iloc[0]
                    for t in tickers], axis=1).mean(axis=1) * start_cash

    def stats(series):
        rets = series.pct_change().dropna()
        years = len(series) / 252
        dd = (series / series.cummax() - 1).min()
        return {"total_return_pct": round((series.iloc[-1] / series.iloc[0] - 1) * 100, 1),
                "cagr_pct": round(((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100, 1),
                "volatility_pct": round(rets.std() * math.sqrt(252) * 100, 1),
                "sharpe": round((rets.mean() * 252 - config.M["risk_free"]) / (rets.std() * math.sqrt(252)), 2) if rets.std() else None,
                "max_drawdown_pct": round(dd * 100, 1)}

    tr = pd.DataFrame(trades)
    summary = {
        "period": f"{eq.index[0].date()} to {eq.index[-1].date()}",
        "tickers": tickers,
        "strategy": stats(eq),
        "benchmark": config.M["benchmark_name"],
        "benchmark_buy_hold": stats(bench),
        "equal_weight_buy_hold": stats(ew),
        "n_trades": len(tr),
        "win_rate_pct": round((tr["pnl"] > 0).mean() * 100, 1) if len(tr) else None,
        "avg_win_pct": round(tr.loc[tr["pnl"] > 0, "return_pct"].mean(), 2) if len(tr) else None,
        "avg_loss_pct": round(tr.loc[tr["pnl"] <= 0, "return_pct"].mean(), 2) if len(tr) else None,
        "exit_reasons": tr["reason"].value_counts().to_dict() if len(tr) else {},
        "avg_capital_invested_pct": round(curve_df["invested"].mean() * 100, 1),
    }
    curves = pd.DataFrame({"Strategy": eq, f"{config.M['benchmark_name']} buy & hold": bench / bench.iloc[0] * start_cash,
                           "Equal-weight buy & hold": ew})
    return {"summary": summary, "curves": curves, "trades": tr}
