"""Download and verify the latest annual reports for the India watchlist.

    python scripts/fetch_annual_reports.py            # download missing reports
    python scripts/fetch_annual_reports.py --force    # re-download everything
    python scripts/fetch_annual_reports.py --check    # verify what is on disk, no network
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight.markets import MARKETS  # noqa: E402
from finsight.report_bot import fetch, report_path, verify  # noqa: E402

if __name__ == "__main__":
    results = []
    for ticker, meta in MARKETS["IN"]["reports"].items():
        if "--check" in sys.argv:
            p = report_path(ticker, meta["fy"])
            res = {"ticker": ticker, "status": "on disk" if p.exists() else "missing",
                   **(verify(p, meta["fy"]) if p.exists() else {})}
        else:
            res = fetch(ticker, meta, force="--force" in sys.argv)
            time.sleep(1)  # be polite between sites
        results.append(res)
        print(json.dumps(res))
    bad = [r for r in results if not r.get("ok")]
    print(f"\n{len(results) - len(bad)}/{len(results)} reports verified" + (f"; check: {[r['ticker'] for r in bad]}" if bad else ""))
