"""Backtest the rule layer + risk engine over the watchlist (default 5 years).

    python scripts/run_backtest.py
    python scripts/run_backtest.py AAPL MSFT NVDA
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finsight import backtest, config  # noqa: E402

if __name__ == "__main__":
    tickers = [a.upper() for a in sys.argv[1:]] or None
    res = backtest.run(tickers)
    print(json.dumps(res["summary"], indent=2))
    out = ROOT / "eval"
    sfx = "" if config.MARKET == "US" else f"_{config.MARKET.lower()}"
    res["curves"].to_csv(out / f"backtest_curves{sfx}.csv")
    res["trades"].to_csv(out / f"backtest_trades{sfx}.csv", index=False)
    (out / f"backtest_summary{sfx}.json").write_text(json.dumps(res["summary"], indent=2))
    print(f"Saved results to {out}")
