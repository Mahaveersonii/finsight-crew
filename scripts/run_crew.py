"""Run the crew from the command line.

    python scripts/run_crew.py AAPL            # analyse + paper-trade one ticker
    python scripts/run_crew.py AAPL MSFT NVDA  # several tickers
    python scripts/run_crew.py --dry-run AAPL  # analyse only, no trade
    MARKET=IN python scripts/run_crew.py ITC   # India (NSE symbol; ".NS" is added for you)
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight import pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    tickers = [a for a in args if not a.startswith("--")] or ["AAPL"]
    for t in tickers:
        res = pipeline.analyze(t, on_event=print, execute_trade=not dry)
        print("\n" + "=" * 70)
        print(f"{t} via {res['model']} in {res['seconds']}s")
        print(json.dumps(res["signal"], indent=2))
        print("Execution:", json.dumps(res["execution"], indent=2, default=str))
