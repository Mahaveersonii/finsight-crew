"""Always-on scheduler (runs as its own container).

  every 15 min   mark-to-market during exchange hours: refresh prices, fire stop-loss /
                 take-profit, snapshot equity (no LLM)
  weekdays after the close     run the full crew over the watchlist (16:30 New York / 16:00 India) (LLM)
  on start       make sure every watchlist ticker's 10-K is in the vector store

    python -m finsight.scheduler            # run forever
    python -m finsight.scheduler --once     # one crew pass over the watchlist, then exit
"""
import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler

from . import broker, config, db, market_clock, pipeline, rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("finsight.scheduler")


def market_open(now=None, grace_min: int = 20) -> bool:
    """Exchange session plus a grace period to capture the close (see market_clock)."""
    return market_clock.is_open(now, grace_min=grace_min)


def mark_to_market_job(force: bool = False):
    db.fail_stale_runs(minutes=15)
    if not force and not market_open():
        return
    if market_clock.is_open():                 # queued orders fill only in the live session
        for ticker, action, outcome in broker.fill_pending():
            log.info("queued %s %s -> %s", action, ticker, outcome)
    res = broker.mark_to_market()               # also logs every automatic stop-loss / take-profit exit
    log.info("mark-to-market: equity=%s exits=%s", res["snapshot"]["equity"], res["exits"])


def crew_job():
    counts, queued, failed = {"BUY": 0, "HOLD": 0, "SELL": 0}, 0, 0
    for ticker in config.WATCHLIST:
        try:
            res = pipeline.analyze(ticker)
            counts[res["signal"]["action"]] = counts.get(res["signal"]["action"], 0) + 1
            queued += res["execution"]["status"] == "pending"
            log.info("%s -> %s (%s)", ticker, res["signal"]["action"], res["execution"]["status"])
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the batch
            failed += 1
            log.exception("crew failed for %s", ticker)
            db.log_event("error", "crew_job", f"{ticker}: {exc}")
    summary = (f"Daily crew run: {len(config.WATCHLIST)} stocks analysed - BUY {counts['BUY']}, "
               f"HOLD {counts['HOLD']}, SELL {counts['SELL']}"
               + (f"; {queued} order(s) queued for the next open" if queued else "")
               + (f"; {failed} failed" if failed else ""))
    db.log_event("crew_batch", "scheduler", summary)
    log.info(summary)


def warm_rag():
    for ticker in config.WATCHLIST:
        try:
            log.info("RAG %s", rag.ingest_ticker(ticker))
        except Exception as exc:  # noqa: BLE001
            log.warning("RAG ingest failed for %s: %s", ticker, exc)


def main():
    log.info("Market %s | DB backend: %s | watchlist: %s", config.MARKET, db.backend(), config.WATCHLIST)
    stale = db.fail_stale_runs(minutes=15)
    if stale:
        log.warning("marked %d interrupted run(s)", stale)
    warm_rag()
    if "--once" in sys.argv:
        crew_job()
        mark_to_market_job(force=True)
        return
    tz, (hh, mm) = config.M["timezone"], config.M["crew_time"]
    sched = BlockingScheduler(timezone=tz)
    sched.add_job(mark_to_market_job, "interval", minutes=15, id="mtm")
    sched.add_job(crew_job, "cron", day_of_week="mon-fri", hour=hh, minute=mm, id="crew")
    mark_to_market_job(force=True)   # one snapshot at start-up, whatever the time
    log.info("Scheduler started")
    sched.start()


if __name__ == "__main__":
    main()
