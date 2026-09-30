"""Wipe paper-trading state (runs, events, signals, trades, positions, snapshots)
for a clean demo. The vector store and market-data cache are kept.

    python scripts/reset_portfolio.py --yes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight import db  # noqa: E402

if __name__ == "__main__":
    if "--yes" not in sys.argv:
        sys.exit("This deletes all runs, signals, trades and positions. Re-run with --yes to confirm.")
    with db.engine().begin() as c:
        for table in (db.events, db.signals, db.trades, db.positions, db.snapshots, db.runs):
            c.execute(table.delete())
    print(f"Reset complete ({db.backend()}). Starting cash restored.")
