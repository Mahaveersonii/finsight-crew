"""Backtest the rule layer + risk engine over the watchlist (default 5 years).

    python scripts/run_backtest.py
    python scripts/run_backtest.py AAPL MSFT NVDA
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finsight import backtest  # noqa: E402

if __name__ == "__main__":
    tickers = [a.upper() for a in sys.argv[1:]] or None
    res = backtest.run(tickers)
    print(json.dumps(res["summary"], indent=2))
    out = ROOT / "eval"
    res["curves"].to_csv(out / "backtest_curves.csv")
    res["trades"].to_csv(out / "backtest_trades.csv", index=False)
    (out / "backtest_summary.json").write_text(json.dumps(res["summary"], indent=2))
    print(f"Saved results to {out}")
