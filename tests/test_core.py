"""Core logic tests - no network, no LLM, isolated SQLite database.

    pytest -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Isolated data dir + SQLite, set before finsight is imported.
_TMP = tempfile.mkdtemp()
os.environ["FINSIGHT_DATA_DIR"] = _TMP
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from finsight import broker, config, db  # noqa: E402
from finsight.crew import extract_json, validate_signal  # noqa: E402
from finsight.rag import chunk_text  # noqa: E402
from finsight.tools import crew_tools as T  # noqa: E402
from finsight.tools import valuation as val  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    with db.engine().begin() as c:
        for t in (db.events, db.signals, db.trades, db.positions, db.snapshots, db.runs):
            c.execute(t.delete())
    T.reset_run(None)
    yield


# --- valuation ---------------------------------------------------------------

def test_dcf_increases_with_growth_and_falls_with_discount():
    lo = val.dcf(100e9, 0.02, 0.10, 1e9)["intrinsic_value_per_share"]
    hi = val.dcf(100e9, 0.10, 0.10, 1e9)["intrinsic_value_per_share"]
    dear = val.dcf(100e9, 0.10, 0.12, 1e9)["intrinsic_value_per_share"]
    assert lo < hi and dear < hi


def test_reverse_dcf_round_trip():
    price = val.dcf(50e9, 0.12, 0.10, 2e9, 10e9, 5e9)["intrinsic_value_per_share"]
    implied = val.implied_growth(price, 50e9, 0.10, 2e9, 10e9, 5e9)
    assert implied == pytest.approx(12.0, abs=0.2)


def test_quant_score_bounds():
    r = {"fcf_yield_pct": 50, "roe_pct": 500, "net_margin_pct": 90, "debt_to_equity": 10}
    q = val.quant_score(r, {"trend": "uptrend", "rsi_14": 55, "return_12m_pct": 400}, 500, 500)
    assert all(0 <= q[k] <= 100 for k in ("value", "quality", "momentum", "composite"))


# --- RAG chunking ------------------------------------------------------------

def test_chunks_overlap_and_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "Risk factor sentence about supply chains. " * 8 for i in range(30))
    chunks = chunk_text(text, size=600, overlap=120)
    assert len(chunks) > 5
    assert all(len(c) <= 600 + 120 + 5 for c in chunks)
    # consecutive chunks share text (overlap) and never start mid-word
    for a, b in zip(chunks, chunks[1:]):
        assert b[:30] in a or b.split("\n\n")[0][:30] in a
        assert b[0].isupper() or b[0].isdigit()


# --- guardrail ---------------------------------------------------------------

GOOD = {"ticker": "AAPL", "action": "buy", "confidence": 72, "time_horizon": "3-6 months",
        "rationale": "x", "key_risks": ["a"], "citations": ["[AAPL 10-K FY2025 · Risk Factors · #4]"]}


def _out(obj):
    return SimpleNamespace(raw=obj if isinstance(obj, str) else "```json\n" + json.dumps(obj) + "\n```")


def test_guardrail_accepts_valid_signal_and_normalises():
    T.RUN["called"] = ["plan_position"]
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · Risk Factors · #4]"}
    ok, value = validate_signal(_out(GOOD))
    sig = json.loads(value)
    assert ok and sig["action"] == "BUY" and sig["confidence"] == 0.72


@pytest.mark.parametrize("bad,msg", [
    ("not json at all", "valid JSON"),
    ({**GOOD, "action": "SHORT"}, "BUY, HOLD or SELL"),
    ({k: v for k, v in GOOD.items() if k != "citations"}, "missing"),
    ({**GOOD, "citations": []}, "citations"),
])
def test_guardrail_rejects_bad_output(bad, msg):
    T.RUN["called"] = ["plan_position"]
    ok, err = validate_signal(_out(bad))
    assert not ok and msg in err


def test_guardrail_rejects_fabricated_citations_and_missing_tool():
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · MD&A · #1]"}
    T.RUN["called"] = ["plan_position"]
    ok, err = validate_signal(_out(GOOD))
    assert not ok and "actually retrieved" in err
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · Risk Factors · #4]"}
    T.RUN["called"] = []
    ok, err = validate_signal(_out(GOOD))
    assert not ok and "plan_position" in err


def test_guardrail_drops_invented_extra_citations():
    T.RUN["called"] = ["plan_position"]
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · Risk Factors · #4]"}
    ok, value = validate_signal(_out({**GOOD, "citations": GOOD["citations"] + ["[AAPL 10-K FY1999 · Made Up · #9]"]}))
    assert ok and json.loads(value)["citations"] == GOOD["citations"]


def test_extract_json_strips_think_tags():
    assert extract_json('<think>hmm {"a": 2}</think> Answer: {"a": 1}') == {"a": 1}


# --- broker / risk engine ----------------------------------------------------

def test_one_percent_risk_sizing_and_position_cap():
    plan = broker.plan_trade("AAA", price=100.0, atr=5.0, sector="Tech")
    assert plan["stop_loss"] == 90.0 and plan["take_profit"] == 120.0
    # 1% of 100k = 1000 risk / 10 per share = 100 shares = 10% of equity (exactly the cap)
    assert plan["shares"] == 100 and plan["capital_at_risk"] == 1000.0
    tight = broker.plan_trade("BBB", price=100.0, atr=0.5, sector="Tech")
    assert tight["shares"] == 100 and tight["binding_constraint"] == "max position size"


def test_low_confidence_buy_is_vetoed():
    res = broker.execute({"ticker": "AAA", "action": "BUY", "price": 100, "confidence": 0.4}, "Tech", 5.0)
    assert res["status"] == "vetoed" and not broker.open_positions()


def test_sector_cap_blocks_fourth_tech_position():
    for t in ("T1", "T2", "T3"):
        assert broker.execute({"ticker": t, "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)["status"] == "executed"
    res = broker.execute({"ticker": "T4", "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)
    assert res["status"] == "vetoed" and "sector" in res["reason"]
    assert broker.execute({"ticker": "E1", "action": "BUY", "price": 100, "confidence": 0.8}, "Energy", 5.0)["status"] == "executed"


def test_sell_without_position_is_noop_and_cash_accounting():
    assert broker.execute({"ticker": "ZZZ", "action": "SELL", "price": 50, "confidence": 0.9}, "X", 1)["status"] == "no_action"
    broker.execute({"ticker": "AAA", "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)
    assert broker.cash() == pytest.approx(config.STARTING_CASH - 100 * 100)
    broker.execute({"ticker": "AAA", "action": "SELL", "price": 110, "confidence": 0.8}, "Tech", 5.0)
    assert broker.cash() == pytest.approx(config.STARTING_CASH + 100 * 10)
    assert not broker.open_positions()


def test_mark_to_market_fires_stop_loss_and_take_profit():
    broker.execute({"ticker": "AAA", "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)
    broker.execute({"ticker": "BBB", "action": "BUY", "price": 100, "confidence": 0.8}, "Health", 5.0)
    prices = {"AAA": 89.0, "BBB": 121.0, "SPY": 500.0}
    out = broker.mark_to_market(price_fn=prices.__getitem__)
    reasons = {t: r for t, r, _ in out["exits"]}
    assert reasons == {"AAA": "stop_loss", "BBB": "take_profit"}
    assert not broker.open_positions()
    assert out["snapshot"]["equity"] == pytest.approx(config.STARTING_CASH - 100 * 11 + 100 * 21)


def test_stale_runs_are_marked_interrupted():
    from datetime import timedelta
    rid = db.start_run("AAA", "m")
    with db.engine().begin() as c:
        c.execute(db.update(db.runs).where(db.runs.c.id == rid).values(started_at=db.now() - timedelta(hours=1)))
    assert db.fail_stale_runs(15) == 1
    assert db.fetch_all("select status from runs")[0]["status"] == "interrupted"


def test_postgres_unreachable_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://x:y@127.0.0.1:1/none")
    monkeypatch.setattr(db, "_engine", None)
    assert db.engine().dialect.name == "sqlite"


def test_llm_chain_skips_models_that_are_not_installed(monkeypatch):
    from finsight import llm
    monkeypatch.setattr(llm, "ollama_models", lambda: {"qwen3:8b"})
    monkeypatch.setattr(config, "PRIMARY_MODEL", "ollama/not-installed")
    monkeypatch.setattr(config, "FALLBACK_MODELS", ["ollama/qwen3:8b"])
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.model_chain() == ["ollama/qwen3:8b"]


def test_unverified_report_citations_are_flagged():
    from finsight.pipeline import _check_report_citations
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · Risk Factors · #4]"}
    rep = "Real [AAPL 10-K FY2025 · Risk Factors · #4] and fake [AAPL 10-K FY2025 · MD&A · #99]."
    out = _check_report_citations(rep, None, lambda m: None)
    assert "#4] and" in out and "#99] ⚠️unverified" in out


def test_tiny_leftover_positions_are_blocked():
    for t in ("T1", "T2"):
        broker.execute({"ticker": t, "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)
    # sector now at 20%; buy ~9.5% more so only ~0.5% of room remains
    broker.execute({"ticker": "T3", "action": "BUY", "price": 95, "confidence": 0.8}, "Tech", 5.0)
    plan = broker.plan_trade("T4", price=40.0, atr=2.0, sector="Tech")
    assert plan["shares"] == 0 and "minimum position" in plan["binding_constraint"]


# --- India market ------------------------------------------------------------

def test_every_india_watchlist_ticker_has_an_annual_report():
    from finsight.markets import MARKETS
    m = MARKETS["IN"]
    assert set(m["watchlist"]) <= set(m["reports"])
    assert all(r["url"].startswith("https://") and r["fy"].isdigit() for r in m["reports"].values())


def test_citation_formats_and_regex_cover_both_markets():
    from finsight.pipeline import _TAG
    from finsight.rag import citation
    us = citation({"form": "10-K", "ticker": "AAPL", "fiscal_year": "2025", "section": "Risk Factors", "chunk": 4})
    india = citation({"form": "AR", "ticker": "ITC.NS", "fiscal_year": "2026", "section": "MD&A", "page": 67, "chunk": 93})
    assert us == "[AAPL 10-K FY2025 · Risk Factors · #4]"
    assert india == "[ITC.NS AR FY2026 · MD&A · p67 #93]"
    assert _TAG.findall(f"x {us} y {india}") == [us, india]


def test_pdf_page_classifier():
    from finsight.pdf_reports import _is_boilerplate, classify
    assert classify("Key risks and mitigation. " + "risk " * 6) == "Risk Management"
    assert classify("Industry outlook: demand growth, revenue and margin expansion, market share gains, "
                    "economic growth, segment profit, EBITDA margin, industry demand outlook.") == "MD&A"
    assert classify("Our purpose and brands.") == "Business & Strategy"
    assert _is_boilerplate("", "Mr. Sanjiv Puri (63), DIN: 00280529, is the Chairman")


def test_india_config_in_a_fresh_process():
    import subprocess
    code = ("from finsight import config as c;"
            "print(c.MARKET, c.normalize_ticker('itc'), c.normalize_ticker('^NSEI'), c.SQLITE_URL.endswith('finsight_in.db'),"
            " c.DATABASE_URL.endswith('/finsight_in'))")
    env = {**os.environ, "MARKET": "IN", "DATABASE_URL": "postgresql+psycopg://u:p@h:5432/finsight"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent)).stdout.split()
    assert out == ["IN", "ITC.NS", "^NSEI", "True", "True"]


def test_guardrail_accepts_citations_without_brackets():
    tag = "[ITC.NS AR FY2026 · MD&A · p67 #92]"
    T.RUN["called"] = ["plan_position"]
    T.RUN["citations"] = {tag}
    ok, value = validate_signal(_out({**GOOD, "citations": ["ITC.NS AR FY2026 · MD&A · p67 #92", " [ITC.NS  AR FY2026 · MD&A · p67 #92 ] "]}))
    assert ok and json.loads(value)["citations"] == [tag, tag]


def test_fact_check_catches_contradicted_numbers(monkeypatch):
    from finsight import crew
    monkeypatch.setattr(crew.T, "valuation", lambda t: {"quant_score": {"composite": 55}, "dcf": {"margin_of_safety_pct": -28.5}})
    monkeypatch.setattr(crew.T, "tech", lambda t: {"trend": "downtrend"})
    bad = crew.fact_check({"ticker": "ITC.NS", "rationale": "Composite score of 72, price below intrinsic value, in an uptrend."})
    assert len(bad) == 3
    assert crew.fact_check({"ticker": "ITC.NS", "rationale": "Composite 55; price is above intrinsic value; downtrend."}) == []
