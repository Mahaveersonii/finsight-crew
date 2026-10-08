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


def test_guardrail_replaces_fabricated_citations_and_rejects_missing_tool():
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · MD&A · #1]"}
    T.RUN["called"] = ["plan_position"]
    ok, value = validate_signal(_out(GOOD))  # invented tag -> replaced by the passage really retrieved
    assert ok and json.loads(value)["citations"] == ["[AAPL 10-K FY2025 · MD&A · #1]"]
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
    assert ok and json.loads(value)["citations"] == [tag]  # same passage twice -> listed once


def test_guardrail_accepts_rewritten_separators_but_not_other_years():
    from finsight.crew import match_citation
    real = ["[AAPL 10-K FY2025 · Risk Factors · #10]"]
    assert match_citation("AAPL 10-K FY2025 | Risk Factors | #10", real) == real[0]
    assert match_citation("[AAPL 10-K FY2025 - Risk Factors - #10]", real) == real[0]
    assert match_citation("[AAPL 10-K FY1999 · Risk Factors · #10]", real) is None
    assert match_citation("[AAPL 10-K FY2025 · MD&A · #10]", real) is None


def test_fact_check_catches_contradicted_numbers(monkeypatch):
    from finsight import crew
    monkeypatch.setattr(crew.T, "valuation", lambda t: {"quant_score": {"composite": 55}, "dcf": {"margin_of_safety_pct": -28.5}})
    monkeypatch.setattr(crew.T, "tech", lambda t: {"trend": "downtrend"})
    bad = crew.fact_check({"ticker": "ITC.NS", "rationale": "Composite score of 72, price below intrinsic value, in an uptrend."})
    assert len(bad) == 3
    assert crew.fact_check({"ticker": "ITC.NS", "rationale": "Composite 55; price is above intrinsic value; downtrend."}) == []


def test_short_price_fetch_never_overwrites_long_history(monkeypatch):
    import pandas as pd
    import yfinance
    from finsight.tools import market_data as md
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    long_df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}, index=idx)

    class Fake:
        def __init__(self, t): pass
        def history(self, period, auto_adjust=True):
            return (long_df if period == "2y" else long_df.iloc[-5:]).tz_localize("UTC")
    monkeypatch.setattr(yfinance, "Ticker", Fake)
    md.get_price_history("ZZZ", "2y")
    md.get_price_history("ZZZ", "5d")          # the 15-minute price check
    class Down:
        def __init__(self, t): raise ConnectionError("outage")
    monkeypatch.setattr(yfinance, "Ticker", Down)
    df = md.get_price_history("ZZZ", "2y")     # Yahoo down during a crew run
    assert len(df) == 500 and df.attrs["stale"]


def test_market_hours_gate():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from finsight.scheduler import market_open
    ny = ZoneInfo("America/New_York")      # tests run with MARKET=US
    assert market_open(datetime(2026, 9, 30, 11, 0, tzinfo=ny))        # Wednesday, session
    assert market_open(datetime(2026, 9, 30, 16, 15, tzinfo=ny))       # grace after close
    assert not market_open(datetime(2026, 9, 30, 20, 0, tzinfo=ny))    # evening
    assert not market_open(datetime(2026, 10, 3, 11, 0, tzinfo=ny))    # Saturday


# --- market clock, order queue, automatic activity ------------------------------

def _pending_signal(ticker, action="BUY", conf=0.8, minutes_ago=60):
    from datetime import timedelta
    with db.engine().begin() as c:
        return c.execute(db.insert(db.signals).values(
            ts=db.now() - timedelta(minutes=minutes_ago), ticker=ticker, action=action, confidence=conf,
            price=100.0, stop_loss=90.0, take_profit=120.0, shares=0, status="pending")).inserted_primary_key[0]


def test_market_clock_status_and_next_open():
    from datetime import datetime
    from finsight import market_clock as mc
    sat = datetime(2026, 10, 3, 11, 0, tzinfo=mc.TZ)
    assert not mc.is_open(sat)
    assert mc.next_open(sat).weekday() == 0                     # Monday
    assert mc.status(sat)["open"] is False and "opens" in mc.status(sat)["detail"]
    wed = datetime(2026, 9, 30, 11, 0, tzinfo=mc.TZ)
    assert mc.is_open(wed) and "closes" in mc.status(wed)["detail"]


def test_buy_is_queued_while_market_closed(monkeypatch):
    from finsight import market_clock
    monkeypatch.setattr(market_clock, "is_open", lambda *a, **k: False)
    res = broker.execute({"ticker": "AAA", "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0,
                         queue_if_closed=True)
    assert res["status"] == "pending" and res["plan"]["shares"] == 100
    assert not broker.open_positions() and not db.fetch_all("select * from trades")
    assert db.fetch_all("select kind from events")[-1]["kind"] == "order_queued"


def test_queued_order_fills_at_open_with_fresh_price(monkeypatch):
    import pandas as pd
    from finsight.tools import market_data as md
    idx = pd.date_range("2026-06-01", periods=70, freq="B")
    hist = pd.DataFrame({"High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    monkeypatch.setattr(md, "get_price_history", lambda t, period="2y", run_id=None: hist)
    monkeypatch.setattr(md, "get_fundamentals", lambda t, run_id=None: {"sector": "Tech"})
    sid = _pending_signal("AAA")
    done = broker.fill_pending(price_fn=lambda t: 105.0)
    assert done == [("AAA", "BUY", "executed")]
    pos = broker.open_positions()[0]
    assert pos["avg_price"] == 105.0                            # filled at the open price, not the stale one
    row = db.fetch_all("select status, price from signals where id = :i", i=sid)[0]
    assert row["status"] == "executed" and row["price"] == 105.0
    assert db.fetch_all("select kind from events")[-1]["kind"] == "order_filled"


def test_newer_decision_supersedes_queued_order():
    sid = _pending_signal("BBB", minutes_ago=120)
    _pending_signal("BBB", action="HOLD", minutes_ago=10)       # newer decision for the same ticker
    with db.engine().begin() as c:
        c.execute(db.update(db.signals).where(db.signals.c.action == "HOLD").values(status="no_action"))
    broker.fill_pending(price_fn=lambda t: 100.0)
    assert db.fetch_all("select status from signals where id = :i", i=sid)[0]["status"] == "superseded"


def test_automatic_exit_is_logged_with_details():
    broker.execute({"ticker": "AAA", "action": "BUY", "price": 100, "confidence": 0.8}, "Tech", 5.0)
    broker.mark_to_market(price_fn={"AAA": 89.0, "SPY": 500.0}.__getitem__)
    ev = db.fetch_all("select kind, name, detail from events where kind = 'auto_exit'")[0]
    assert ev["name"] == "stop_loss" and "Sold 100 AAA at 89.00" in ev["detail"] and "P&L -1,100.00" in ev["detail"]


def test_guardrail_uses_decision_submitted_through_json_tool():
    T.RUN["called"] = ["plan_position"]
    T.RUN["citations"] = {"[AAPL 10-K FY2025 · Risk Factors · #4]"}
    T.RUN["submitted"] = dict(GOOD)
    ok, value = validate_signal(_out("Decision submitted."))
    T.RUN["submitted"] = None
    assert ok and json.loads(value)["action"] == "BUY"


# --- Filing Change Analyst ---------------------------------------------------

def test_cut_section_skips_table_of_contents():
    from finsight.filing_changes import SECTIONS, cut_section
    toc = "Item 1A. Risk Factors 5\nItem 1B. Unresolved Staff Comments 17\nItem 2. Properties 18\n" + "x " * 200
    body = "Item 1A. Risk Factors\n" + "Real risk text. " * 40 + "\nItem 1B. Unresolved Staff Comments\nnone\n"
    out = cut_section(toc + "\n" + body, *SECTIONS["Risk Factors"])
    assert out.startswith("Item 1A. Risk Factors\nReal risk text") and "Unresolved" not in out


def test_compare_classifies_unchanged_edited_new_and_removed():
    import numpy as np
    from finsight.filing_changes import compare
    base = "The Company depends on component suppliers in Asia and any disruption could hurt results "
    old = [base + "materially.", "Interest rates may rise and reduce demand for financed purchases across markets.",
           "Legacy product line X may be discontinued which would reduce revenue from older customers."]
    new = [base + "materially.", base + "materially and quickly, including new tariffs on imported parts.",
           "New online safety laws may require age verification for all app store users worldwide today."]
    out = compare(new, old, embed=lambda t: np.ones((len(t), 4)) / 2)  # equal meaning scores: words decide
    assert out["unchanged"] == 1
    assert [d["n"] for d in out["edited"]] == [2]
    assert [d["n"] for d in out["added"]] == [3]
    assert {d["n"] for d in out["removed"]} == {2, 3}


def test_lazy_prices_similarity_and_explanation_guardrail():
    from finsight.filing_changes import _check, cosine_similarity, tag
    assert cosine_similarity("risk of tariffs on parts", "risk of tariffs on parts") == 1.0
    assert cosine_similarity("risk of tariffs", "weather was sunny") == 0.0
    real = tag("AAPL", "2025", "Risk Factors", "new", 84)
    raw = json.dumps({"headline": "h", "concern": "high", "tone": "t", "removed_or_softened": [],
                      "new_risks": [{"risk": "online safety", "citations": [real]},
                                    {"risk": "invented", "citations": ["[AAPL 10-K FY2025 · Risk Factors · new ¶999]"]}]})
    ok, out = _check(raw, {real})
    assert ok and out["concern"] == "High" and [r["risk"] for r in out["new_risks"]] == ["online safety"]



def test_price_history_drops_blank_rows_before_the_open(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from finsight.tools import market_data as md
    idx = pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07"])
    raw = pd.DataFrame({"Open": [1.0, 2.0, np.nan], "High": [1.0, 2.0, np.nan], "Low": [1.0, 2.0, np.nan],
                        "Close": [1.0, 2.0, np.nan], "Volume": [10, 20, 30]}, index=idx)

    class FakeTicker:
        def __init__(self, t):
            pass

        def history(self, **kw):
            return raw.copy()
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(md, "_cache_path", lambda name: tmp_path / name)
    df = md.get_price_history("TEST", period="5d")
    assert len(df) == 2 and df["Close"].iloc[-1] == 2.0
