"""Persistence layer: PostgreSQL with automatic SQLite fallback.

Tables
  runs            one row per crew run (ticker, model used, status, duration)
  events          fine-grained log: tool calls, fallbacks, errors (feeds Grafana)
  signals         final structured BUY/HOLD/SELL decision from the crew
  trades          paper-trade fills
  positions       current open positions
  snapshots       portfolio equity over time (feeds Grafana / Streamlit)
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String, Table, Text,
    create_engine, insert, select, text, update, delete,
)

from . import config

log = logging.getLogger(__name__)
metadata = MetaData()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


runs = Table(
    "runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(16), index=True),
    Column("started_at", DateTime, default=now),
    Column("finished_at", DateTime),
    Column("model", String(128)),
    Column("status", String(32)),          # running | success | failed
    Column("duration_s", Float),
    Column("attempts", Integer),
    Column("report", Text),                # analyst report (markdown)
)

events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=now, index=True),
    Column("run_id", Integer, index=True),
    Column("kind", String(32), index=True),  # tool_call | llm_fallback | data_fallback | guardrail_retry | error | risk_veto
    Column("name", String(128)),
    Column("detail", Text),
    Column("duration_ms", Float),
)

signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=now, index=True),
    Column("run_id", Integer),
    Column("ticker", String(16), index=True),
    Column("action", String(8)),           # BUY | HOLD | SELL
    Column("confidence", Float),
    Column("price", Float),
    Column("stop_loss", Float),
    Column("take_profit", Float),
    Column("shares", Integer),
    Column("status", String(16)),          # executed | vetoed | no_action
    Column("rationale", Text),
    Column("citations", Text),             # JSON list
    Column("raw", Text),
)

trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=now, index=True),
    Column("ticker", String(16), index=True),
    Column("side", String(4)),             # BUY | SELL
    Column("shares", Integer),
    Column("price", Float),
    Column("reason", String(64)),          # signal | stop_loss | take_profit
    Column("signal_id", Integer),
    Column("realized_pnl", Float),
)

positions = Table(
    "positions", metadata,
    Column("ticker", String(16), primary_key=True),
    Column("shares", Integer),
    Column("avg_price", Float),
    Column("stop_loss", Float),
    Column("take_profit", Float),
    Column("sector", String(64)),
    Column("last_price", Float),
    Column("opened_at", DateTime, default=now),
)

snapshots = Table(
    "snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=now, index=True),
    Column("cash", Float),
    Column("positions_value", Float),
    Column("equity", Float),
    Column("benchmark", Float),            # SPY close, for comparison
)

_engine = None


def engine():
    """Return a cached engine. Try Postgres first, fall back to SQLite."""
    global _engine
    if _engine is not None:
        return _engine
    if config.DATABASE_URL:
        try:
            eng = create_engine(config.DATABASE_URL, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(text("select 1"))
            _engine = eng
            log.info("Using PostgreSQL")
        except Exception as exc:  # noqa: BLE001
            log.warning("PostgreSQL unavailable (%s) - falling back to SQLite", exc)
    if _engine is None:
        _engine = create_engine(config.SQLITE_URL)
    metadata.create_all(_engine)
    return _engine


def backend() -> str:
    return engine().dialect.name


# --- small helpers ---------------------------------------------------------

def start_run(ticker: str, model: str) -> int:
    with engine().begin() as c:
        res = c.execute(insert(runs).values(ticker=ticker, model=model, status="running", started_at=now()))
        return res.inserted_primary_key[0]


def finish_run(run_id: int, **values):
    values.setdefault("finished_at", now())
    with engine().begin() as c:
        c.execute(update(runs).where(runs.c.id == run_id).values(**values))


def log_event(kind: str, name: str, detail="", run_id=None, duration_ms=None):
    if not isinstance(detail, str):
        detail = json.dumps(detail, default=str)
    try:
        with engine().begin() as c:
            c.execute(insert(events).values(
                ts=now(), run_id=run_id, kind=kind, name=name,
                detail=detail[:4000], duration_ms=duration_ms,
            ))
    except Exception as exc:  # noqa: BLE001 - logging must never break a run
        log.warning("could not log event: %s", exc)


def fetch_all(query, **params):
    with engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(query if not isinstance(query, str) else text(query), params)]


def recent_signals(ticker: str, limit: int = 5):
    q = select(signals).where(signals.c.ticker == ticker).order_by(signals.c.ts.desc()).limit(limit)
    return fetch_all(q)


__all__ = [
    "engine", "backend", "runs", "events", "signals", "trades", "positions", "snapshots",
    "start_run", "finish_run", "log_event", "fetch_all", "recent_signals", "insert", "select",
    "update", "delete", "now",
]
