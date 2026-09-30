"""Always-on scheduler (runs as its own container).

  every 15 min   mark-to-market: refresh prices, fire stop-loss / take-profit, snapshot equity  (no LLM)
  weekdays 16:30 US/Eastern   run the full crew over the watchlist                              (LLM)
  on start       make sure every watchlist ticker's 10-K is in the vector store

    python -m finsight.scheduler            # run forever
    python -m finsight.scheduler --once     # one crew pass over the watchlist, then exit
"""
import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler

from . import broker, config, db, pipeline, rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("finsight.scheduler")


def mark_to_market_job():
    db.fail_stale_runs(minutes=15)
    res = broker.mark_to_market()
    log.info("mark-to-market: equity=%s exits=%s", res["snapshot"]["equity"], res["exits"])
    for ticker, reason, pnl in res["exits"]:
        db.log_event("auto_exit", reason, f"{ticker} closed, P&L {pnl:.2f}")


def crew_job():
    for ticker in config.WATCHLIST:
        try:
            res = pipeline.analyze(ticker)
            log.info("%s -> %s (%s)", ticker, res["signal"]["action"], res["execution"]["status"])
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the batch
            log.exception("crew failed for %s", ticker)
            db.log_event("error", "crew_job", f"{ticker}: {exc}")


def warm_rag():
    for ticker in config.WATCHLIST:
        try:
            log.info("RAG %s", rag.ingest_ticker(ticker))
        except Exception as exc:  # noqa: BLE001
            log.warning("RAG ingest failed for %s: %s", ticker, exc)


def main():
    log.info("DB backend: %s | watchlist: %s", db.backend(), config.WATCHLIST)
    stale = db.fail_stale_runs(minutes=15)
    if stale:
        log.warning("marked %d interrupted run(s)", stale)
    warm_rag()
    if "--once" in sys.argv:
        crew_job()
        mark_to_market_job()
        return
    sched = BlockingScheduler(timezone="America/New_York")
    sched.add_job(mark_to_market_job, "interval", minutes=15, id="mtm")
    sched.add_job(crew_job, "cron", day_of_week="mon-fri", hour=16, minute=30, id="crew")
    mark_to_market_job()
    log.info("Scheduler started")
    sched.start()


if __name__ == "__main__":
    main()
