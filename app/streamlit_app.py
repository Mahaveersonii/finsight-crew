"""FinSight Crew - Streamlit control room.

    streamlit run app/streamlit_app.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight import broker, config, db, market_clock, pipeline, rag  # noqa: E402
from finsight.llm import model_chain  # noqa: E402

MK = config.M
CUR = MK["symbol"]
DOC = "10-K" if MK["corpus"] == "sec" else "Annual Report"
BENCH = MK["benchmark_name"]
FLAG = {"US": "🇺🇸", "IN": "🇮🇳"}

st.set_page_config(page_title=f"FinSight Crew · {MK['name']}", page_icon="📈", layout="wide")


TZ_LABEL = {"America/New_York": "New York time", "Asia/Kolkata": "IST"}[MK["timezone"]]
STATUS_WORDS = {"executed": "✅ traded", "no_action": "no trade", "vetoed": "⛔ blocked", "analysis_only": "analysis only",
                "pending": "⏳ waiting for market open", "superseded": "replaced by a newer decision", "expired": "expired"}
ACTIVITY_KINDS = ("auto_exit", "order_queued", "order_filled", "order_cancelled", "crew_batch")
ACTIVITY_ICON = {"auto_exit": "⚡", "order_queued": "⏳", "order_filled": "✅", "order_cancelled": "✖️", "crew_batch": "🤖"}


def recent_activity(limit=20):
    """Things the system did on its own: automatic exits, queued / filled orders, daily crew runs."""
    rows = db.fetch_all("select ts, kind, detail from events where kind in "
                        "('auto_exit','order_queued','order_filled','order_cancelled','crew_batch') "
                        "order by id desc limit :n", n=limit)
    for r in rows:
        r["when"] = pd.Timestamp(r["ts"]).tz_localize("UTC").tz_convert(MK["timezone"]).strftime("%d %b %H:%M")
    return rows


def local_time(df, *cols):
    """Stored timestamps are UTC; show them in the market's own timezone."""
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC").dt.tz_convert(MK["timezone"]).dt.strftime("%d %b %H:%M")
    return df


def money_short(x):
    """Compact form for metric tiles: ₹1.00 Cr / ₹10.50 L in India, $100.6K / $1.20M in the US."""
    if x is None:
        return "–"
    if MK["currency"] == "INR":
        if abs(x) >= 1e7:
            return f"{CUR}{x / 1e7:.2f} Cr"
        if abs(x) >= 1e5:
            return f"{CUR}{x / 1e5:.2f} L"
        return money(x)
    if abs(x) >= 1e6:
        return f"{CUR}{x / 1e6:.2f}M"
    if abs(x) >= 1e4:
        return f"{CUR}{x / 1e3:.1f}K"
    return money(x)


def money(x, dec=0):
    """$1,234,567 for the US; ₹12,34,567 (lakh/crore grouping) for India."""
    if x is None:
        return "–"
    if MK["currency"] != "INR":
        return f"{CUR}{x:,.{dec}f}"
    neg, x = x < 0, abs(x)
    whole, frac = f"{x:.{dec}f}".split(".") if dec else (f"{x:.0f}", "")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    body = ",".join(groups + [tail]) if groups else tail
    return f"{'-' if neg else ''}{CUR}{body}{'.' + frac if frac else ''}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 FinSight Crew")
    st.caption(f"Autonomous financial research & paper trading · CrewAI + Ollama + RAG over {MK['doc_name']}s")
    st.markdown(f"**Market:** {FLAG[config.MARKET]} {MK['name']} · {MK['currency']} · benchmark {BENCH}")
    _ms = market_clock.status()
    (st.success if _ms["open"] else st.error)(f"{'🟢' if _ms['open'] else '🔴'} **{_ms['label']}**, {_ms['detail']}")
    _act = recent_activity(3)
    if _act:
        st.markdown("**Latest automatic activity**")
        for a in _act:
            st.caption(f"{ACTIVITY_ICON[a['kind']]} {a['when']} · {a['detail']}")
    for code, url in config.APP_URLS.items():
        if code != config.MARKET:
            from finsight.markets import MARKETS
            st.link_button(f"Switch to {FLAG[code]} {MARKETS[code]['name']} →", url, width="stretch")
    chain = model_chain()
    st.markdown("**LLM fallback chain**")
    for i, m in enumerate(chain):
        st.markdown(f"{'🟢' if i == 0 else '⚪'} `{m}`")
    if not chain:
        st.error("No LLM reachable. Start Ollama.")
    st.markdown(f"**Database:** `{db.backend()}`")
    idx = rag.indexed_tickers()
    st.markdown(f"**Vector store:** {sum(idx.values())} chunks · {len(idx)} companies")
    st.divider()
    st.markdown("**Risk limits**")
    st.markdown(
        f"- {config.RISK_PER_TRADE:.0%} equity risked per trade\n"
        f"- ≤ {config.MAX_POSITION_PCT:.0%} per position\n"
        f"- ≤ {config.MAX_SECTOR_PCT:.0%} per sector\n"
        f"- Stop = entry − {config.ATR_STOP_MULT:g}×ATR(14)\n"
        f"- Take-profit = 2R\n- BUY needs confidence ≥ {config.MIN_CONFIDENCE}"
    )

@st.cache_data(ttl=600, show_spinner="Loading market data…")
def _overview(tk):
    """Cached per ticker for 10 min. Deliberately does not touch crew_tools.RUN, so browsing
    here can never reset the state of a crew run in progress."""
    from finsight.tools import market_data as md, valuation as val
    df = md.get_price_history(tk)
    try:
        bench = md.get_price_history(MK["benchmark"])
    except Exception:  # noqa: BLE001
        bench = None
    rep = val.full_report(md.get_fundamentals(tk), md.get_fcf_history(tk), df, bench)
    return rep, val.technicals(df), df.reset_index()


tab_run, tab_port, tab_rag, tab_bt, tab_ops = st.tabs(
    ["🧠 Run the Crew", "💼 Portfolio", f"📚 Ask the {DOC}", "📈 Backtest & RAG Eval", "🔍 Agent Ops"])


# ---------------------------------------------------------------------------
# Run the crew
# ---------------------------------------------------------------------------
with tab_run:
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = config.normalize_ticker(c1.text_input(f"{MK['name']} ticker (NSE symbol)" if config.MARKET == "IN" else "US ticker",
                                                   value=config.WATCHLIST[0].replace(MK["ticker_suffix"], ""), max_chars=14))
    trade = c2.toggle("Execute paper trade", value=True)
    go_btn = c3.button("▶ Run analysis", type="primary", width="stretch", disabled=not chain)

    if ticker:
        try:
            v, tc, px_df = _overview(ticker)
            m = st.columns(6)
            m[0].metric("Price", money(v['price']))
            m[1].metric("P/E (ttm)", v["ratios"]["pe_trailing"])
            m[2].metric("EV/EBITDA", v["ratios"]["ev_to_ebitda"])
            m[3].metric("FCF yield", f"{v['ratios']['fcf_yield_pct']:.1f}%" if v["ratios"]["fcf_yield_pct"] else "–")
            m[4].metric("12m return", f"{tc['return_12m_pct']:+.0f}%" if tc["return_12m_pct"] is not None else "–")
            m[5].metric("Quant score", v["quant_score"]["composite"])
            px_df["SMA50"] = px_df["Close"].rolling(50).mean()
            px_df["SMA200"] = px_df["Close"].rolling(200).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=px_df["Date"], y=px_df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=px_df["Date"], y=px_df["SMA50"], name="SMA 50", line={"dash": "dot"}))
            fig.add_trace(go.Scatter(x=px_df["Date"], y=px_df["SMA200"], name="SMA 200", line={"dash": "dash"}))
            fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend={"orientation": "h"})
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load data for {ticker}: {exc}")

    if go_btn:
        with st.status(f"Crew analysing {ticker}…", expanded=True) as status:
            log_box = st.empty()
            lines, out = [], {}

            # The crew runs on a worker thread and only appends messages; this (script) thread
            # redraws the log every second, so every tool call shows up live.
            def work():
                try:
                    out["res"] = pipeline.analyze(ticker, on_event=lines.append, execute_trade=trade)
                except Exception as exc:  # noqa: BLE001
                    out["err"] = exc

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            t0 = time.time()
            while worker.is_alive():
                log_box.code("\n".join(lines[-25:] + [f"… working ({time.time() - t0:.0f}s)"]), language=None)
                time.sleep(1)
            log_box.code("\n".join(lines[-25:]), language=None)
            if "res" in out:
                res = out["res"]
                status.update(label=f"Done in {res['seconds']}s on {res['model']}", state="complete")
                st.session_state["last_result"] = res
            else:
                status.update(label=f"Failed: {out.get('err')}", state="error")

    res = st.session_state.get("last_result")
    if res:
        sig, ex = res["signal"], res["execution"]
        colour = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[sig["action"]]
        st.markdown(f"## :{colour}[{sig['action']}] {sig['ticker']} · confidence {sig['confidence']:.0%}")
        k = st.columns(5)
        k[0].metric("Entry", money(sig['price'], 2))
        k[1].metric("Stop-loss", money(sig['stop_loss'], 2))
        k[2].metric("Take-profit", money(sig['take_profit'], 2))
        k[3].metric("Shares", (ex.get("plan") or {}).get("shares", 0) if ex["status"] == "executed" else 0)
        k[4].metric("Execution", STATUS_WORDS.get(ex["status"], ex["status"]))
        if ex["status"] == "pending":
            st.info(f"⏳ {ex['reason']}. The scheduler will buy at the opening price, with the stop-loss and "
                    "size recalculated from that price.")
        elif ex["status"] == "no_action":
            st.info("No trade placed: the decision was HOLD." if sig["action"] == "HOLD" else f"No trade placed: {ex.get('reason')}")
        elif ex.get("reason"):
            st.warning(f"Risk engine: {ex['reason']}")
        if sig.get("policy_override"):
            st.warning(f"Policy check: {sig['policy_override']}")
        st.markdown(f"**Rationale.** {sig['rationale']}")
        st.markdown("**Key risks:** " + " · ".join(sig.get("key_risks", [])))
        st.markdown(f"**{DOC} citations:** " + " ".join(f"`{c}`" for c in sig.get("citations", [])))
        with st.expander("📄 Analyst report"):
            st.markdown(res["analyst_report"])
        with st.expander("📊 Data brief"):
            st.markdown(res["data_brief"])
        with st.expander("🧾 Raw signal JSON"):
            st.json(sig)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
with tab_port:
    if st.button("↻ Mark to market now"):
        out = broker.mark_to_market()
        exits = ", ".join(f"{t} ({r.replace('_', '-')}, P&L {money(pnl)})" for t, r, pnl in out["exits"]) or "none"
        st.success(f"Equity {money(out['snapshot']['equity'], 2)} · automatic exits: {exits}")
    summ = broker.portfolio_summary()
    start = config.STARTING_CASH
    c = st.columns(4)
    c[0].metric("Equity", money_short(summ['equity']), f"{(summ['equity'] / start - 1):.2%}", help=money(summ['equity'], 2))
    c[1].metric("Cash", money_short(summ['cash']), help=money(summ['cash'], 2))
    c[2].metric("Open positions", len(summ["positions"]))
    c[3].metric("Invested", f"{1 - summ['cash'] / summ['equity']:.0%}")

    snaps = pd.DataFrame(db.fetch_all("select ts, equity, benchmark from snapshots order by ts"))
    if len(snaps) > 1 and snaps["benchmark"].notna().any():
        snaps["ts"] = pd.to_datetime(snaps["ts"]).dt.tz_localize("UTC").dt.tz_convert(MK["timezone"])
        b0 = snaps["benchmark"].dropna().iloc[0]
        snaps["Portfolio"] = snaps["equity"] / start * 100
        snaps[BENCH] = snaps["benchmark"] / b0 * 100
        fig = px.line(snaps, x="ts", y=["Portfolio", BENCH], title=f"Growth of 100 (portfolio vs {BENCH})",
                      labels={"ts": f"Time ({TZ_LABEL})", "value": "Value (start = 100)", "variable": ""})
        fig.update_layout(legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    pos = pd.DataFrame(db.fetch_all("select * from positions"))
    if not pos.empty:
        pos["value"] = pos["shares"] * pos["last_price"]
        pos["unrealised_pnl"] = (pos["last_price"] - pos["avg_price"]) * pos["shares"]
        left.markdown("#### Open positions")
        left.dataframe(pos[["ticker", "shares", "avg_price", "last_price", "stop_loss", "take_profit", "unrealised_pnl", "sector"]],
                       hide_index=True, width="stretch")
        right.plotly_chart(px.pie(pos, values="value", names="ticker", hole=0.5, title="Allocation"), width="stretch")
    else:
        left.info("No open positions yet - run the crew on a few tickers.")

    st.markdown(f"#### ⚡ Automatic activity · times in {TZ_LABEL}")
    st.caption("Everything the system did on its own: stop-loss / take-profit sales, orders queued while the "
               "market was closed and filled at the open, and the daily crew runs.")
    act = recent_activity(20)
    if act:
        for a in act:
            st.markdown(f"{ACTIVITY_ICON[a['kind']]} **{a['when']}** · {a['detail']}")
    else:
        st.info("Nothing yet. Automatic sales, queued orders and daily crew runs will appear here.")

    pend = pd.DataFrame(db.fetch_all(
        "select ts, ticker, action, confidence, price as decided_at_price from signals where status = 'pending' order by ts"))
    if not pend.empty:
        st.markdown("#### ⏳ Orders waiting for the market to open")
        st.dataframe(local_time(pend, "ts"), hide_index=True, width="stretch")

    st.markdown(f"#### Signals · times in {TZ_LABEL}")
    sigs = pd.DataFrame(db.fetch_all(
        "select ts, ticker, action, confidence, price, stop_loss, shares, status, rationale from signals order by ts desc limit 50"))
    if sigs.empty:
        st.info("No signals yet.")
    else:
        sigs["status"] = sigs["status"].map(lambda v: STATUS_WORDS.get(v, v))
        st.dataframe(local_time(sigs, "ts"), hide_index=True, width="stretch")
    st.markdown(f"#### Trades · times in {TZ_LABEL}")
    trades_df = pd.DataFrame(db.fetch_all(
        "select ts, ticker, side, shares, price, reason, realized_pnl from trades order by ts desc limit 50"))
    if trades_df.empty:
        st.info("No trades yet. Every signal so far was HOLD or was blocked by the risk rules.")
    else:
        st.dataframe(local_time(trades_df, "ts"), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
with tab_rag:
    st.markdown(f"Ask questions answered **only** from the company's latest {MK['doc_name']}, with citations.")
    idx = rag.indexed_tickers()
    c1, c2 = st.columns([1, 3])
    rt = c1.selectbox("Company", sorted(idx) or config.WATCHLIST)
    new_t = config.normalize_ticker(c1.text_input("…or index a new ticker"))
    if new_t and c1.button(f"Index {DOC}"):
        with st.spinner(f"Downloading and embedding the {new_t} {DOC}…"):
            try:
                st.success(rag.ingest_ticker(new_t))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not index {new_t}: {exc}")
    q = c2.text_input("Question", value="What are the biggest risks management highlights?")
    if c2.button("Ask", type="primary") and q:
        with st.spinner("Retrieving and answering…"):
            out = rag.answer(rt, q)
        st.markdown(out["answer"])
        st.markdown("##### Retrieved passages")
        for h in out["sources"]:
            with st.expander(f"{h['citation']} · score {h['score']} (semantic {h['semantic']})"):
                st.write(h["text"])
                if h.get("url"):
                    link = h["url"] + (f"#page={h['page']}" if h.get("page") else "")
                    where = f"page {h['page']} of the report" if h.get("page") else "the filing on SEC.gov"
                    st.markdown(f"[Open {where}]({link})")


# ---------------------------------------------------------------------------
# Backtest + RAG evaluation
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Backtesting 5 years of the watchlist…")
def _backtest(tickers):
    from finsight import backtest
    r = backtest.run(list(tickers))
    return r["summary"], r["curves"], r["trades"]


with tab_bt:
    st.markdown("#### Backtest: momentum rules + the fund's risk engine (no LLM)")
    st.caption("Entry weekly when close > SMA50 > SMA200 and 40 ≤ RSI ≤ 70; exits on 2×ATR stop, 2R take-profit or "
               "close < SMA200; 1% risk sizing, ≤10% per position. Agents and fundamentals are not backtested "
               "(that would need historical fundamentals - look-ahead bias).")
    extra = (["AMZN", "GOOGL", "META", "KO", "PG", "CVX"] if config.MARKET == "US"
             else ["TCS.NS", "HINDUNILVR.NS", "MARUTI.NS", "LT.NS", "ULTRACEMCO.NS", "ONGC.NS"])
    bt_t = st.multiselect("Universe", config.WATCHLIST + extra, default=config.WATCHLIST)
    if bt_t and st.button("Run backtest"):
        st.session_state["bt"] = _backtest(tuple(bt_t))
    if "bt" in st.session_state:
        summ, curves, trs = st.session_state["bt"]
        rows = []
        for name, label in (("strategy", "Strategy"), ("benchmark_buy_hold", f"{BENCH} buy & hold"),
                            ("equal_weight_buy_hold", "Equal-weight buy & hold")):
            rows.append({"": label, **summ[name]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        c = st.columns(5)
        c[0].metric("Trades", summ["n_trades"])
        c[1].metric("Win rate", f"{summ['win_rate_pct']}%")
        c[2].metric("Avg win / loss", f"{summ['avg_win_pct']}% / {summ['avg_loss_pct']}%")
        c[3].metric("Avg capital invested", f"{summ['avg_capital_invested_pct']}%")
        c[4].metric("Exits", ", ".join(f"{k}: {v}" for k, v in summ["exit_reasons"].items()))
        st.plotly_chart(px.line(curves, title=f"Equity curves · {summ['period']}"), width="stretch")
        with st.expander("Trade list"):
            st.dataframe(trs, hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### RAG retrieval evaluation")
    ev_path = Path(__file__).resolve().parent.parent / "eval" / (
        "rag_eval_results.json" if config.MARKET == "US" else f"rag_eval_results_{config.MARKET.lower()}.json")
    if ev_path.exists():
        ev_res = json.loads(ev_path.read_text())
        st.caption(f"{ev_res['n_questions']} paraphrased questions over {', '.join(ev_res.get('tickers', ['AAPL', 'NVDA', 'JNJ']))} "
                   f"{DOC}s. "
                   "A hit = a retrieved chunk contains the ground-truth fact. Run `python eval/run_rag_eval.py` to refresh.")
        mdf = pd.DataFrame([{"configuration": k, **v["metrics"]} for k, v in ev_res["results"].items()])
        st.dataframe(mdf, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(mdf.melt(id_vars="configuration", var_name="metric"), x="metric", y="value",
                               color="configuration", barmode="group", title="Vector-only vs hybrid re-ranking"),
                        width="stretch")
    else:
        st.info("Run `python eval/run_rag_eval.py` to generate evaluation results.")


# ---------------------------------------------------------------------------
# Agent ops
# ---------------------------------------------------------------------------
with tab_ops:
    runs = local_time(pd.DataFrame(db.fetch_all(
        "select id, ticker, started_at, model, status, attempts, duration_s from runs order by id desc limit 50")), "started_at")
    ev = local_time(pd.DataFrame(db.fetch_all(
        "select ts, run_id, kind, name, duration_ms, detail from events order by id desc limit 500")), "ts")
    c = st.columns(5)
    c[0].metric("Crew runs", len(runs))
    done = runs[runs["status"] != "running"] if len(runs) else runs
    c[1].metric("Success rate", f"{(done['status'] == 'success').mean():.0%}" if len(done) else "–")
    c[2].metric("Avg run time", f"{runs['duration_s'].mean():.0f}s" if len(runs) and runs["duration_s"].notna().any() else "–")
    c[3].metric("Tool calls", int((ev["kind"] == "tool_call").sum()) if len(ev) else 0)
    c[4].metric("Recoveries", int(ev["kind"].isin(["llm_fallback", "data_fallback", "guardrail_retry", "risk_override"]).sum()) if len(ev) else 0)
    if len(ev):
        l, r = st.columns(2)
        tc = ev[ev["kind"] == "tool_call"].groupby("name")["duration_ms"].agg(["count", "mean"]).reset_index()
        l.plotly_chart(px.bar(tc, x="name", y="count", title="Tool calls by tool"), width="stretch")
        r.plotly_chart(px.histogram(ev, x="kind", title="Events by type"), width="stretch")
        st.markdown(f"#### Event log (tool calls, fallbacks, guardrails, vetoes) · times in {TZ_LABEL}")
        st.dataframe(ev, hide_index=True, width="stretch", height=320)
    st.markdown(f"#### Runs · times in {TZ_LABEL}")
    st.dataframe(runs, hide_index=True, width="stretch")
    if len(runs):
        rid = st.selectbox("Open full report for run", runs["id"])
        rep = db.fetch_all("select report from runs where id = :i", i=int(rid))
        if rep and rep[0]["report"]:
            st.markdown(rep[0]["report"])
