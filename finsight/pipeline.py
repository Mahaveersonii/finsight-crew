"""End-to-end orchestration for one ticker:

  preflight data -> ensure 10-K indexed -> crew (with LLM fallback chain)
  -> guardrail-validated JSON signal -> deterministic risk engine -> paper fill
  -> persistence (runs / signals / trades / snapshots / events)
"""
import json
import logging
import re
import threading
import time

from . import broker, db, rag
from .crew import build_crew, extract_json
from .db import insert, signals, update
from .llm import make_llm, model_chain
from .tools import crew_tools as T

log = logging.getLogger(__name__)

# Tool state (crew_tools.RUN) is per process, and a local 8B model serves one request
# at a time anyway, so allow exactly one crew run per process.
_RUN_LOCK = threading.Lock()


class CrewBusy(RuntimeError):
    pass


def analyze(ticker: str, on_event=None, execute_trade: bool = True) -> dict:
    if not _RUN_LOCK.acquire(blocking=False):
        raise CrewBusy("Another crew run is in progress - wait for it to finish.")
    try:
        return _analyze(ticker, on_event, execute_trade)
    finally:
        _RUN_LOCK.release()


_TAG = re.compile(r"\[[A-Z.\-]+ 10-K FY\d{4} · [^\]]+ · #\d+\]")


def _check_report_citations(report: str, run_id, emit) -> str:
    """Flag any 10-K citation in the analyst report that was not actually retrieved in this run."""
    real = T.RUN["citations"]
    bad = sorted({c for c in _TAG.findall(report) if c not in real})
    for c in bad:
        report = report.replace(c, f"{c} ⚠️unverified")
    if bad:
        db.log_event("citation_unverified", "analyst_report", ", ".join(bad), run_id)
        emit(f"⚠️ {len(bad)} analyst citation(s) not found in retrieved passages - flagged in the report")
    return report


def _analyze(ticker: str, on_event=None, execute_trade: bool = True) -> dict:
    ticker = ticker.strip().upper()
    emit = on_event or (lambda msg: log.info(msg))
    chain = model_chain()
    if not chain:
        raise RuntimeError("No LLM available: start Ollama or set GROQ_API_KEY / GEMINI_API_KEY")

    run_id = db.start_run(ticker, chain[0])
    T.reset_run(run_id)
    T.RUN["on_event"] = on_event
    t0 = time.time()

    # 1. Preflight: fetch data once so every agent sees the same numbers.
    emit(f"📡 Fetching market data for {ticker}")
    try:
        fund, tech = T.fundamentals(ticker), T.tech(ticker)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, status="failed", duration_s=time.time() - t0, report=str(exc))
        raise
    if ticker not in rag.indexed_tickers():
        emit(f"📚 Indexing latest 10-K for {ticker} into the vector store")
        emit(f"   → {rag.ingest_ticker(ticker)}")

    # 2. Crew, with model fallback.
    result, used_model, attempts = None, None, 0
    for model in chain:
        attempts += 1
        emit(f"🤖 Running crew on {model}")
        try:
            crew = build_crew(ticker, make_llm(model))
            result = crew.kickoff()
            used_model = model
            break
        except Exception as exc:  # noqa: BLE001
            db.log_event("llm_fallback", model, f"{type(exc).__name__}: {exc}", run_id)
            emit(f"⚠️ {model} failed ({type(exc).__name__}) — falling back to next model")
    if result is None:
        db.finish_run(run_id, status="failed", attempts=attempts, duration_s=time.time() - t0)
        raise RuntimeError("All models in the fallback chain failed")

    outputs = [t.raw for t in result.tasks_output]
    outputs[1] = _check_report_citations(outputs[1], run_id, emit)
    sig = extract_json(outputs[-1])

    # 3. Enforce the fund's decision rules deterministically (the LLM can misapply them).
    score = (T.valuation(ticker).get("quant_score") or {}).get("composite")
    held = any(p["ticker"] == ticker for p in broker.open_positions())
    if sig["action"] == "BUY" and score is not None and score < 70:
        db.log_event("risk_veto", "policy:composite<70", f"{ticker} BUY -> HOLD (composite {score})", run_id)
        emit(f"🛡️ Policy check: BUY needs composite >= 70, got {score} -> HOLD")
        sig["action"], sig["policy_override"] = "HOLD", f"BUY downgraded: composite {score} < 70"
    if sig["action"] == "SELL" and not held:
        sig["action"], sig["policy_override"] = "HOLD", "SELL ignored: no position (long-only)"

    # 4. The risk engine, not the LLM, owns price / stop / size.
    price = tech["last_close"]
    sig.update(ticker=ticker, price=price)
    sector = fund.get("sector") or "Unknown"
    plan = broker.plan_trade(ticker, price, tech["atr_14"], sector)
    llm_stop = sig.get("stop_loss")
    if sig["action"] == "BUY" and isinstance(llm_stop, (int, float)) and abs(llm_stop - plan["stop_loss"]) > 0.01 * price:
        db.log_event("risk_override", "stop_loss", f"LLM proposed {llm_stop}, risk engine set {plan['stop_loss']}", run_id)
        emit(f"🛡️ Risk engine overrode LLM stop-loss {llm_stop} → {plan['stop_loss']}")
    sig.update(stop_loss=plan["stop_loss"], take_profit=plan["take_profit"])

    with db.engine().begin() as c:
        signal_id = c.execute(insert(signals).values(
            ts=db.now(), run_id=run_id, ticker=ticker, action=sig["action"], confidence=sig["confidence"],
            price=price, stop_loss=sig["stop_loss"], take_profit=sig["take_profit"], shares=0, status="pending",
            rationale=sig.get("rationale"), citations=json.dumps(sig.get("citations", [])), raw=json.dumps(sig),
        )).inserted_primary_key[0]

    execution = {"status": "analysis_only", "reason": "paper trading switched off for this run"}
    if not execute_trade:
        with db.engine().begin() as c:
            c.execute(update(signals).where(signals.c.id == signal_id).values(status="analysis_only"))
    else:
        execution = broker.execute(sig, sector, tech["atr_14"], run_id, signal_id)
        shares = (execution.get("plan") or {}).get("shares") or execution.get("shares") or 0
        with db.engine().begin() as c:
            c.execute(update(signals).where(signals.c.id == signal_id).values(
                status=execution["status"], shares=shares if execution["status"] == "executed" else 0))
        broker.mark_to_market()
    emit(f"✅ {ticker}: {sig['action']} (confidence {sig['confidence']:.2f}) → {execution['status']}")

    report = "\n\n---\n\n".join(outputs)
    db.finish_run(run_id, status="success", model=used_model, attempts=attempts,
                  duration_s=round(time.time() - t0, 1), report=report)
    return {"run_id": run_id, "model": used_model, "signal": sig, "execution": execution,
            "data_brief": outputs[0], "analyst_report": outputs[1], "seconds": round(time.time() - t0, 1)}
