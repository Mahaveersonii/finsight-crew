"""FinSight Crew - Streamlit control room.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight import broker, config, db, pipeline, rag  # noqa: E402
from finsight.llm import model_chain  # noqa: E402

st.set_page_config(page_title="FinSight Crew", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 FinSight Crew")
    st.caption("Autonomous financial research & paper trading · CrewAI + Ollama + RAG over SEC 10-Ks")
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
    tech = val.technicals(df)
    return val.valuation_report(md.get_fundamentals(tk), md.get_fcf_history(tk), tech), tech, df.reset_index()


tab_run, tab_port, tab_rag, tab_bt, tab_ops = st.tabs(
    ["🧠 Run the Crew", "💼 Portfolio", "📚 Ask the 10-K", "📈 Backtest & RAG Eval", "🔍 Agent Ops"])


# ---------------------------------------------------------------------------
# Run the crew
# ---------------------------------------------------------------------------
with tab_run:
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.text_input("US ticker", value="MSFT", max_chars=8).strip().upper()
    trade = c2.toggle("Execute paper trade", value=True)
    go_btn = c3.button("▶ Run analysis", type="primary", width="stretch", disabled=not chain)

    if ticker:
        try:
            v, tc, px_df = _overview(ticker)
            m = st.columns(6)
            m[0].metric("Price", f"${v['price']:,.0f}")
            m[1].metric("P/E (ttm)", v["ratios"]["pe_trailing"])
            m[2].metric("EV/EBITDA", v["ratios"]["ev_to_ebitda"])
            m[3].metric("FCF yield", f"{v['ratios']['fcf_yield_pct']:.1f}%" if v["ratios"]["fcf_yield_pct"] else "–")
            m[4].metric("12m return", f"{tc['return_12m_pct']}%")
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
            lines = []

            def on_event(msg):
                lines.append(msg)
                try:  # only works on the script thread; worker-thread messages show on the next update
                    log_box.code("\n".join(lines[-25:]), language=None)
                except Exception:  # noqa: BLE001
                    pass

            try:
                res = pipeline.analyze(ticker, on_event=on_event, execute_trade=trade)
                status.update(label=f"Done in {res['seconds']}s on {res['model']}", state="complete")
                st.session_state["last_result"] = res
            except Exception as exc:  # noqa: BLE001
                status.update(label=f"Failed: {exc}", state="error")

    res = st.session_state.get("last_result")
    if res:
        sig, ex = res["signal"], res["execution"]
        colour = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[sig["action"]]
        st.markdown(f"## :{colour}[{sig['action']}] {sig['ticker']} · confidence {sig['confidence']:.0%}")
        k = st.columns(5)
        k[0].metric("Entry", f"${sig['price']}")
        k[1].metric("Stop-loss", f"${sig['stop_loss']}")
        k[2].metric("Take-profit", f"${sig['take_profit']}")
        k[3].metric("Shares", (ex.get("plan") or {}).get("shares", 0) if ex["status"] == "executed" else 0)
        k[4].metric("Execution", ex["status"])
        if ex.get("reason"):
            st.info(f"Risk engine: {ex['reason']}")
        st.markdown(f"**Rationale.** {sig['rationale']}")
        st.markdown("**Key risks:** " + " · ".join(sig.get("key_risks", [])))
        st.markdown("**10-K citations:** " + " ".join(f"`{c}`" for c in sig.get("citations", [])))
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
        st.success(f"Equity ${out['snapshot']['equity']:,.2f} · exits: {out['exits'] or 'none'}")
    summ = broker.portfolio_summary()
    start = config.STARTING_CASH
    c = st.columns(4)
    c[0].metric("Equity", f"${summ['equity']:,.0f}", f"{(summ['equity'] / start - 1):.2%}")
    c[1].metric("Cash", f"${summ['cash']:,.0f}")
    c[2].metric("Open positions", len(summ["positions"]))
    c[3].metric("Invested", f"{1 - summ['cash'] / summ['equity']:.0%}")

    snaps = pd.DataFrame(db.fetch_all("select ts, equity, benchmark from snapshots order by ts"))
    if len(snaps) > 1 and snaps["benchmark"].notna().any():
        b0 = snaps["benchmark"].dropna().iloc[0]
        snaps["Portfolio"] = snaps["equity"] / start * 100
        snaps["SPY"] = snaps["benchmark"] / b0 * 100
        st.plotly_chart(px.line(snaps, x="ts", y=["Portfolio", "SPY"], title="Growth of 100 (portfolio vs SPY)"),
                        width="stretch")

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

    st.markdown("#### Signals")
    sigs = pd.DataFrame(db.fetch_all(
        "select ts, ticker, action, confidence, price, stop_loss, shares, status, rationale from signals order by ts desc limit 50"))
    st.dataframe(sigs, hide_index=True, width="stretch")
    st.markdown("#### Trades")
    st.dataframe(pd.DataFrame(db.fetch_all("select * from trades order by ts desc limit 50")), hide_index=True,
                 width="stretch")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
with tab_rag:
    st.markdown("Ask questions answered **only** from the company's latest SEC 10-K, with citations.")
    idx = rag.indexed_tickers()
    c1, c2 = st.columns([1, 3])
    rt = c1.selectbox("Company", sorted(idx) or ["AAPL"])
    new_t = c1.text_input("…or index a new ticker").strip().upper()
    if new_t and c1.button("Index 10-K"):
        with st.spinner(f"Downloading and embedding {new_t} 10-K…"):
            st.success(rag.ingest_ticker(new_t))
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
                    st.markdown(f"[Open filing on SEC.gov]({h['url']})")


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
    bt_t = st.multiselect("Universe", config.WATCHLIST + ["AMZN", "GOOGL", "META", "KO", "PG", "CVX"], default=config.WATCHLIST)
    if bt_t and st.button("Run backtest"):
        st.session_state["bt"] = _backtest(tuple(bt_t))
    if "bt" in st.session_state:
        summ, curves, trs = st.session_state["bt"]
        rows = []
        for name in ("strategy", "spy_buy_hold", "equal_weight_buy_hold"):
            rows.append({"": name.replace("_", " "), **summ[name]})
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
    ev_path = Path(__file__).resolve().parent.parent / "eval" / "rag_eval_results.json"
    if ev_path.exists():
        ev_res = json.loads(ev_path.read_text())
        st.caption(f"{ev_res['n_questions']} paraphrased questions over AAPL / NVDA / JNJ 10-Ks. "
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
    runs = pd.DataFrame(db.fetch_all(
        "select id, ticker, started_at, model, status, attempts, duration_s from runs order by id desc limit 50"))
    ev = pd.DataFrame(db.fetch_all("select ts, run_id, kind, name, duration_ms, detail from events order by id desc limit 500"))
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
        st.markdown("#### Event log (tool calls, fallbacks, guardrails, vetoes)")
        st.dataframe(ev, hide_index=True, width="stretch", height=320)
    st.markdown("#### Runs")
    st.dataframe(runs, hide_index=True, width="stretch")
    if len(runs):
        rid = st.selectbox("Open full report for run", runs["id"])
        rep = db.fetch_all("select report from runs where id = :i", i=int(rid))
        if rep and rep[0]["report"]:
            st.markdown(rep[0]["report"])
