"""FinSight Crew - Streamlit trading desk.

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
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402
from finsight import broker, config, db, market_clock, pipeline, rag  # noqa: E402
from finsight.llm import model_chain  # noqa: E402
from ui import C  # noqa: E402

MK = config.M
CUR = MK["symbol"]
DOC = "10-K" if MK["corpus"] == "sec" else "Annual Report"
BENCH = MK["benchmark_name"]
FLAG = {"US": "🇺🇸", "IN": "🇮🇳"}

st.set_page_config(page_title=f"FinSight Crew · {MK['name']}", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)


TZ_LABEL = {"America/New_York": "New York time", "Asia/Kolkata": "IST"}[MK["timezone"]]
STATUS_WORDS = {"executed": "✅ traded", "no_action": "no trade", "vetoed": "⛔ blocked", "analysis_only": "analysis only",
                "pending": "⏳ waiting for market open", "superseded": "replaced by a newer decision", "expired": "expired"}
ACTIVITY_ICON = {"auto_exit": "⚡", "order_queued": "⏳", "order_filled": "✅", "order_cancelled": "✖️", "crew_batch": "🤖"}
RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}


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
    """Compact form for tiles: ₹1.00 Cr / ₹10.50 L in India, $100.6K / $1.20M in the US."""
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


def signed_money(x, dec=0):
    return "–" if x is None else ("+" if x >= 0 else "−") + money(abs(x), dec)


def pct(x, dec=1, sign=True):
    return "–" if x is None else f"{x:+.{dec}f}%" if sign else f"{x:.{dec}f}%"


def html(s):
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading market data…")
def _overview(tk):
    """Cached per ticker for 5 min (charts, valuation, scores). Deliberately does not touch crew_tools.RUN, so browsing
    here can never reset the state of a crew run in progress."""
    from finsight.tools import market_data as md, valuation as val
    df = md.get_price_history(tk)
    try:
        bench = md.get_price_history(MK["benchmark"])
    except Exception:  # noqa: BLE001
        bench = None
    rep = val.full_report(md.get_fundamentals(tk), md.get_fcf_history(tk), df, bench)
    return rep, val.technicals(df), df.reset_index()


LIVE_SECONDS = 60  # how often prices refresh on screen while the exchange is open


@st.cache_data(ttl=LIVE_SECONDS, show_spinner=False)
def _quote(tk):
    """Latest price and today's change. A 5-day daily history includes today's live bar while the
    exchange is open, so this is the current traded price (Yahoo may delay it by a few minutes)."""
    from finsight.tools import market_data as md
    close = md.get_price_history(tk, period="5d")["Close"].dropna()
    return float(close.iloc[-1]), float((close.iloc[-1] / close.iloc[-2] - 1) * 100)


@st.cache_data(ttl=600, show_spinner=False)
def _spark(tk):
    from finsight.tools import market_data as md
    return md.get_price_history(tk)["Close"].dropna().tail(60).round(2).tolist()


def quotes(tickers):
    out = {}
    for t in tickers:
        try:
            out[t] = _quote(t)
        except Exception:  # noqa: BLE001 - one missing quote must not break the page
            pass
    return out


def short(tk):
    return tk.replace(MK["ticker_suffix"], "") if MK["ticker_suffix"] else tk


# ---------------------------------------------------------------------------
# Header, ticker tape and sidebar
# ---------------------------------------------------------------------------
chain = model_chain()
summ = broker.portfolio_summary()
start_cash = config.STARTING_CASH
model_name = chain[0].split("/", 1)[-1] if chain else None
LIVE = market_clock.is_open()


@st.fragment(run_every=LIVE_SECONDS if LIVE else None)
def live_bar():
    """Header and ticker tape. While the exchange is open this part reloads itself every minute,
    without re-running the rest of the page."""
    ms = market_clock.status()
    eq = broker.portfolio_summary()["equity"]
    clock = market_clock.now().strftime("%a %H:%M:%S") + (" · live" if ms["open"] else "")
    html(ui.header(MK["name"], FLAG[config.MARKET], ms, model_name, money(eq),
                   f"{signed_money(eq - start_cash)} ({pct((eq / start_cash - 1) * 100, 2)})", eq - start_cash, clock))
    q = quotes(config.WATCHLIST + [MK["benchmark"]])
    html(ui.tape([(BENCH if t == MK["benchmark"] else short(t), money(v[0], 2), v[1]) for t, v in q.items()]))


live_bar()

with st.sidebar:
    html('<div class="fs-brand"><div class="fs-logo">FS</div><div>FinSight Crew<small>Control panel</small></div></div>')
    for code, url in config.APP_URLS.items():
        if code != config.MARKET:
            from finsight.markets import MARKETS
            st.link_button(f"Switch to {FLAG[code]} {MARKETS[code]['name']} →", url, width="stretch")
    html('<div class="fs-side-h">AI model chain</div>')
    html("".join(ui.pill(m.split("/", 1)[-1], "acc" if i == 0 else "") + " " for i, m in enumerate(chain))
         or ui.pill("No model reachable", "warn"))
    idx = rag.indexed_tickers()
    html('<div class="fs-side-h">Data</div>')
    html(ui.cards([{"label": "Database", "value": db.backend()},
                   {"label": "Report chunks", "value": f"{sum(idx.values()):,}", "sub": f"{len(idx)} companies indexed"}]))
    html('<div class="fs-side-h">Risk rules</div>')
    html(ui.chips([f"{config.RISK_PER_TRADE:.0%} risk per trade", f"≤ {config.MAX_POSITION_PCT:.0%} per position",
                   f"≤ {config.MAX_SECTOR_PCT:.0%} per sector", f"Stop = entry − {config.ATR_STOP_MULT:g}×ATR",
                   "Target = 2R", f"BUY needs confidence ≥ {config.MIN_CONFIDENCE}", "BUY needs score ≥ 70", "Long only"]))
    html('<div class="fs-side-h">Latest automatic activity</div>')
    act3 = recent_activity(4)
    html(ui.timeline([(ACTIVITY_ICON[a["kind"]], a["when"], a["detail"]) for a in act3]) if act3
         else ui.empty("Nothing yet"))


tab_desk, tab_scan, tab_chg, tab_port, tab_rag, tab_bt, tab_ops = st.tabs(
    ["Trade desk", "Scanner", "What changed", "Portfolio", f"Ask the {DOC}", "Backtest lab", "Agent ops"])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def price_chart(px_df, tk, days):
    d = px_df.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d = d.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=short(tk),
                                 increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
                                 decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"])), 1, 1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SMA50"], name="SMA 50", line=dict(color=C["accent"], width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SMA200"], name="SMA 200", line=dict(color=C["warn"], width=1.6, dash="dot")), 1, 1)
    vol_col = [C["up"] if c >= o else C["down"] for o, c in zip(d["Open"], d["Close"])]
    fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"], marker_color=vol_col, opacity=0.45, name="Volume", showlegend=False), 2, 1)

    held = next((p for p in db.fetch_all("select * from positions") if p["ticker"] == tk), None)
    if held:
        for y, name, col in ((held["avg_price"], "Entry", C["text"]), (held["stop_loss"], "Stop", C["down"]),
                             (held["take_profit"], "Target", C["up"])):
            if y:
                fig.add_hline(y=y, line=dict(color=col, width=1, dash="dash"), row=1, col=1,
                              annotation_text=f"{name} {money(y, 2)}", annotation_font_color=col, annotation_position="top left")
    trs = pd.DataFrame(db.fetch_all("select ts, side, price, shares from trades where ticker = :t", t=tk))
    if not trs.empty:
        trs["ts"] = pd.to_datetime(trs["ts"]).dt.normalize()
        trs = trs[trs["ts"] >= d["Date"].min()]
        for side, sym, col in (("buy", "triangle-up", C["up"]), ("sell", "triangle-down", C["down"])):
            s = trs[trs["side"] == side]
            if len(s):
                fig.add_trace(go.Scatter(x=s["ts"], y=s["price"], mode="markers", name=f"{side.title()} fills",
                                         marker=dict(symbol=sym, size=13, color=col, line=dict(color=C["bg"], width=1)),
                                         text=[f"{side} {n} @ {money(p, 2)}" for n, p in zip(s["shares"], s["price"])],
                                         hoverinfo="text"), 1, 1)
    fig.update_layout(height=470, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=30, b=8))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text=None, side="right")
    return fig


def valuation_band(v):
    sc = (v.get("dcf") or {}).get("scenarios") or {}
    pts = [("DCF bear", sc.get("bear"), C["down"]), ("DCF base", sc.get("base"), C["warn"]), ("DCF bull", sc.get("bull"), C["up"]),
           ("Analyst target", (v.get("analyst_consensus") or {}).get("target_mean_price"), C["accent2"]),
           ("Price now", v.get("price"), C["text"])]
    pts = [p for p in pts if p[1]]
    fig = go.Figure()
    xs = [p[1] for p in pts]
    fig.add_shape(type="line", x0=min(xs), x1=max(xs), y0=0, y1=0, line=dict(color=C["line"], width=6))
    for i, (name, x, col) in enumerate(pts):
        fig.add_trace(go.Scatter(x=[x], y=[0], mode="markers+text", name=name,
                                 marker=dict(size=16 if name == "Price now" else 12, color=col,
                                             symbol="diamond" if name == "Price now" else "circle",
                                             line=dict(color=C["bg"], width=2)),
                                 text=[f"{name}<br>{money(x, 0)}"], textposition="top center" if i % 2 == 0 else "bottom center",
                                 textfont=dict(size=11), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=170, margin=dict(l=30, r=30, t=10, b=10), title=None,
                      yaxis=dict(visible=False, range=[-1.2, 1.2]), xaxis=dict(showgrid=False, tickprefix=CUR))
    return fig


# ---------------------------------------------------------------------------
# Trade desk
# ---------------------------------------------------------------------------
with tab_desk:
    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2], vertical_alignment="bottom")
    pick = c1.selectbox("Stock", [short(t) for t in config.WATCHLIST] + ["Other…"],
                        help="Pick from the watchlist, or choose Other to type any ticker.")
    if pick == "Other…":
        pick = c1.text_input("Ticker", value="", max_chars=14, placeholder="e.g. AMZN" if config.MARKET == "US" else "e.g. TCS")
    ticker = config.normalize_ticker(pick) if pick else ""
    rng = c2.segmented_control("Range", list(RANGES), default="1Y")
    trade = c3.toggle("Paper-trade the decision", value=True)
    go_btn = c4.button("Run the agents", type="primary", width="stretch", disabled=not (chain and ticker))

    if ticker:
        try:
            v, tc, px_df = _overview(ticker)
            qs = v["quant_score"]

            @st.fragment(run_every=LIVE_SECONDS if LIVE else None)
            def live_price():
                q = _quote(ticker)
                html(ui.cards([{"label": "Last price" + (" · live" if market_clock.is_open() else " · last close"),
                                "value": money(q[0], 2), "sub": f"{pct(q[1], 2)} today", "tone": ui.tone_of(q[1])}]))

            pc, rest = st.columns([1, 4])
            with pc:
                live_price()
            rest.markdown(ui.cards([
                {"label": "Trend", "value": tc["trend"].replace("trend", " trend").title(), "sub": f"RSI {tc['rsi_14']} · ATR {tc['atr_14']}",
                 "tone": "fs-up" if tc["trend"] == "uptrend" else "fs-down" if tc["trend"] == "downtrend" else ""},
                {"label": "P/E · EV/EBITDA", "value": f"{v['ratios']['pe_trailing'] or '–'} · {v['ratios']['ev_to_ebitda'] or '–'}",
                 "sub": f"FCF yield {pct(v['ratios']['fcf_yield_pct'], 1, False)}"},
                {"label": "12-month return", "value": pct(tc["return_12m_pct"], 0), "tone": ui.tone_of(tc["return_12m_pct"]),
                 "sub": f"max drawdown {pct(tc['max_drawdown_1y_pct'], 0, False)}"},
                {"label": "Quant score", "value": f"{qs['composite']} / 100",
                 "tone": "fs-up" if qs["composite"] >= 70 else "fs-warn" if qs["composite"] >= 50 else "fs-down",
                 "sub": f"value {qs['value']} · quality {qs['quality']} · momentum {qs['momentum']}"},
            ]), unsafe_allow_html=True)
            left, right = st.columns([2.3, 1])
            with left:
                st.plotly_chart(price_chart(px_df, ticker, RANGES[rng or "1Y"]), width="stretch",
                                config={"displayModeBar": False})
            with right:
                st.plotly_chart(ui.gauge(qs["composite"]), width="stretch", config={"displayModeBar": False})
                pillars = go.Figure(go.Bar(
                    x=[qs["value"], qs["quality"], qs["momentum"]], y=["Value", "Quality", "Momentum"], orientation="h",
                    marker_color=[C["accent"], C["accent2"], C["up"]], text=[qs["value"], qs["quality"], qs["momentum"]],
                    textposition="outside"))
                pillars.update_layout(height=170, xaxis=dict(range=[0, 115], visible=False), margin=dict(l=8, r=8, t=8, b=8),
                                      title=None)
                st.plotly_chart(pillars, width="stretch", config={"displayModeBar": False})
            html(ui.section("Where the price sits", "DCF scenarios and the analyst target against today's price"))
            st.plotly_chart(valuation_band(v), width="stretch", config={"displayModeBar": False})
            flags = v.get("risk_flags") or []
            if flags:
                html(ui.chips([f"⚠ {f}" for f in flags]))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load data for {ticker}: {exc}")

    if go_btn:
        html(ui.section("Agents at work", f"{short(ticker)} · {model_name}"))
        step_box, log_box = st.empty(), st.empty()
        lines, out = [], {}

        # The crew runs on a worker thread and only appends messages; this (script) thread
        # redraws the steps and the log every second, so every tool call shows up live.
        def work():
            try:
                out["res"] = pipeline.analyze(ticker, on_event=lines.append, execute_trade=trade)
            except Exception as exc:  # noqa: BLE001
                out["err"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        t0 = time.time()
        while worker.is_alive():
            step_box.markdown(ui.stepper(ui.stage_from_log(lines)), unsafe_allow_html=True)
            log_box.markdown(ui.terminal(lines[-18:] + [f"… working ({time.time() - t0:.0f}s)"]), unsafe_allow_html=True)
            time.sleep(1)
        step_box.markdown(ui.stepper(ui.stage_from_log(lines)),
                          unsafe_allow_html=True)
        log_box.markdown(ui.terminal(lines[-18:]), unsafe_allow_html=True)
        if "res" in out:
            st.session_state["last_result"] = out["res"]
            st.toast(f"{short(ticker)}: {out['res']['signal']['action']} in {out['res']['seconds']}s", icon="✅")
        else:
            st.error(f"The run failed: {out.get('err')}")

    res = st.session_state.get("last_result")
    if res:
        sig, ex = res["signal"], res["execution"]
        html(ui.section("Latest decision", f"{short(sig['ticker'])} · {res['model'].split('/', 1)[-1]} · {res['seconds']}s"))
        shares = (ex.get("plan") or {}).get("shares", 0) if ex["status"] == "executed" else 0
        notes = []
        if ex["status"] == "pending":
            notes.append(("warn", f"{ex['reason']}. The scheduler buys at the next open, re-sizing from that price."))
        elif ex["status"] == "no_action":
            notes.append(("", "No trade: the decision was HOLD." if sig["action"] == "HOLD" else f"No trade: {ex.get('reason')}"))
        elif ex.get("reason") and ex["status"] != "analysis_only":
            notes.append(("warn", f"Risk engine: {ex['reason']}"))
        if sig.get("policy_override"):
            notes.append(("warn", f"Policy check: {sig['policy_override']}"))
        html(f"""<div class="fs-decision">
          <div class="fs-action"><span class="big {sig['action']}">{sig['action']}</span>
            <div><div style="font-size:1.15rem;font-weight:700">{ui.e(short(sig['ticker']))} · confidence {sig['confidence']:.0%}</div>
            <div class="fs-muted">{ui.e(sig.get('time_horizon', ''))} · {ui.e(STATUS_WORDS.get(ex['status'], ex['status']))}
            {f' · {shares} shares' if shares else ''}</div></div></div>
          {ui.rr_bar(sig['stop_loss'], sig['price'], sig['take_profit'], lambda x: money(x, 2))}
          {''.join(ui.pill(t, tone, dot=False) for tone, t in notes)}
          <div>{ui.e(sig['rationale'])}</div>
          <div><div class="fs-side-h">Key risks</div>{ui.chips(sig.get('key_risks', []))}</div>
          <div><div class="fs-side-h">{DOC} evidence</div>{ui.chips(sig.get('citations', []), 'cite')}</div>
        </div>""")
        e1, e2, e3 = st.columns(3)
        with e1.expander("Analyst report"):
            st.markdown(res["analyst_report"])
        with e2.expander("Data brief"):
            st.markdown(res["data_brief"])
        with e3.expander("Raw signal JSON"):
            st.json(sig)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
with tab_scan:
    html(ui.section("Watchlist scanner", "every stock scored with the same model the agents use, refreshed every 10 minutes"))
    last_sig = {}
    for r in db.fetch_all("select ticker, action, ts from signals order by id desc"):
        last_sig.setdefault(r["ticker"], f"{r['action']} · {pd.Timestamp(r['ts']).strftime('%d %b')}")
    rows = []
    for tk in config.WATCHLIST:
        try:
            v, tc, _ = _overview(tk)
            q = _quote(tk)
            rows.append({"Stock": short(tk), "Price": q[0], "1D %": round(q[1], 2), "60-day trend": _spark(tk),
                         "1M %": tc["return_1m_pct"], "12M %": tc["return_12m_pct"], "Trend": tc["trend"], "RSI": tc["rsi_14"],
                         "Score": v["quant_score"]["composite"], "Value": v["quant_score"]["value"],
                         "Quality": v["quant_score"]["quality"], "Momentum": v["quant_score"]["momentum"],
                         "vs DCF %": (v.get("dcf") or {}).get("margin_of_safety_pct"),
                         "Last decision": last_sig.get(tk, "–")})
        except Exception as exc:  # noqa: BLE001
            rows.append({"Stock": short(tk), "Trend": f"data unavailable: {exc}"})
    scan = pd.DataFrame(rows)
    if "Score" in scan:
        scan = scan.sort_values("Score", ascending=False, na_position="last")
        buy_ready = scan[(scan["Score"] >= 70)]["Stock"].tolist()
        html(ui.cards([
            {"label": "Stocks tracked", "value": len(scan)},
            {"label": "Score ≥ 70 (BUY zone)", "value": len(buy_ready), "sub": ", ".join(buy_ready) or "none today",
             "tone": "fs-up" if buy_ready else ""},
            {"label": "In an uptrend", "value": int((scan["Trend"] == "uptrend").sum())},
            {"label": "Best 12-month", "value": f"{scan.loc[scan['12M %'].idxmax(), 'Stock']} {pct(scan['12M %'].max(), 0)}"
             if scan["12M %"].notna().any() else "–"},
        ]))
    st.dataframe(scan, hide_index=True, width="stretch", column_config={
        "Price": st.column_config.NumberColumn(format=f"{CUR}%.2f"),
        "1D %": st.column_config.NumberColumn(format="%+.2f%%"),
        "1M %": st.column_config.NumberColumn(format="%+.1f%%"),
        "12M %": st.column_config.NumberColumn(format="%+.1f%%"),
        "vs DCF %": st.column_config.NumberColumn(format="%+.0f%%", help="Margin of safety against the base-case DCF value"),
        "60-day trend": st.column_config.LineChartColumn(width="medium"),
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d", help="BUY needs 70 or more"),
        "Value": None, "Quality": None, "Momentum": None,
    })
    if "Score" in scan and scan["Score"].notna().any():
        sc = scan.dropna(subset=["Score"])
        fig = px.scatter(sc, x="Value", y="Momentum", size="Quality", color="Score", text="Stock", size_max=46,
                         range_color=[0, 100], color_continuous_scale=[[0, C["down"]], [0.5, C["warn"]], [1, C["up"]]],
                         title="Score map: value vs momentum (bubble size = quality)")
        fig.update_traces(textposition="middle center", textfont=dict(color=C["bg"], size=11),
                          marker=dict(opacity=0.95, line=dict(color=C["bg"], width=1)))
        fig.update_layout(height=420, xaxis=dict(range=[-5, 105]), yaxis=dict(range=[-5, 105]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# What changed (Filing Change Analyst)
# ---------------------------------------------------------------------------
CONCERN_TONE = {"High": "down", "Medium": "warn", "Low": "up"}

with tab_chg:
    html(ui.section("What changed in the annual report",
                    "this year's 10-K against last year's · research basis: Lazy Prices, Journal of Finance 2020"))
    if config.MARKET != "US":
        html(ui.empty("The year-on-year comparison needs last year's annual reports too. It works for US 10-Ks now; "
                      "India follows once the FY2025 reports are added."))
    else:
        from finsight import filing_changes as fc
        c1, c2, c3 = st.columns([2.2, 1.1, 1.1], vertical_alignment="bottom")
        ct = config.normalize_ticker(c1.selectbox("Company", [short(t) for t in config.WATCHLIST], key="chg_ticker"))
        if c2.button("Compare reports", type="primary", width="stretch", disabled=not chain):
            with st.spinner(f"Reading the last two {ct} 10-Ks and asking the Filing Change Analyst…"):
                try:
                    fc.analyse(ct)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not compare {ct}: {exc}")
        if c3.button("Compare all 6", width="stretch", disabled=not chain):
            bar = st.progress(0.0, text="Comparing reports…")
            for i, t in enumerate(config.WATCHLIST):
                try:
                    fc.analyse(t)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{t}: {exc}")
                bar.progress((i + 1) / len(config.WATCHLIST), text=f"Compared {t}")
            bar.empty()

        done = {t: r for t in config.WATCHLIST if (r := fc.cached(t))}
        if done:
            rows = []
            for t, r in done.items():
                sec = r["sections"].get("Risk Factors")
                if not sec:
                    continue
                total = sec["unchanged"] + len(sec["edited"]) + len(sec["added"])
                for kind, n in (("Unchanged", sec["unchanged"]), ("Edited", len(sec["edited"])), ("New", len(sec["added"]))):
                    rows.append({"Company": short(t), "Paragraphs": kind, "Share": round(n / total * 100, 1), "Count": n,
                                 "Concern": (r.get("summary") or {}).get("concern", "–")})
            if rows:
                fig = px.bar(pd.DataFrame(rows), y="Company", x="Share", color="Paragraphs", orientation="h",
                             hover_data=["Count", "Concern"], title="Risk Factors: share of this year's paragraphs",
                             color_discrete_map={"Unchanged": C["line"], "Edited": C["warn"], "New": C["down"]})
                fig.update_layout(height=110 + 42 * len(done), barmode="stack", yaxis_title=None, xaxis_title=None,
                                  legend_title_text="", legend=dict(orientation="h", y=-0.22, x=0),
                                  margin=dict(l=8, r=8, t=40, b=50), xaxis=dict(range=[0, 100], ticksuffix="%"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        res = fc.cached(ct)
        if not res:
            html(ui.empty(f"No comparison for {short(ct)} yet. Press Compare reports: code lines up every paragraph "
                          "of the last two 10-Ks, then the Filing Change Analyst explains what is new."))
        else:
            sm = res.get("summary") or {}
            rf, md_ = res["sections"].get("Risk Factors", {}), res["sections"].get("MD&A", {})

            def counts(sec):
                return f"{len(sec.get('added', []))} new · {len(sec.get('edited', []))} edited · {len(sec.get('removed', []))} removed"

            html(ui.cards([
                {"label": "Compared", "value": f"FY{res['fy_new']} vs FY{res['fy_old']}", "sub": f"filed {res['filed_new']}"},
                {"label": "Risk Factors", "value": f"{rf.get('similarity', 0):.3f}", "sub": counts(rf)},
                {"label": "MD&A", "value": f"{md_.get('similarity', 0):.3f}", "sub": counts(md_)},
                {"label": "Concern", "value": sm.get("concern", "–"),
                 "tone": {"High": "fs-down", "Medium": "fs-warn", "Low": "fs-up"}.get(sm.get("concern"), ""),
                 "sub": f"judged by {str(sm.get('model') or '–').split('/', 1)[-1]}"},
            ]))
            st.caption("Similarity is the word-count cosine used in the Lazy Prices study: 1.000 means identical wording.")
            if sm:
                risks = "".join(
                    f'<div style="margin-top:10px"><b>{ui.e(r.get("risk", ""))}</b><div class="fs-muted">{ui.e(r.get("why_it_matters", ""))}</div>'
                    f'{ui.chips(r.get("citations", []), "cite")}</div>' for r in sm.get("new_risks", []))
                gone = "".join(f'<div style="margin-top:8px">{ui.e(r.get("item", ""))}{ui.chips(r.get("citations", []), "cite")}</div>'
                               for r in sm.get("removed_or_softened", []))
                html(f"""<div class="fs-decision">
                  <div class="fs-action">{ui.pill(f"Concern: {sm.get('concern', '–')}", CONCERN_TONE.get(sm.get('concern'), ''))}
                    <div style="font-size:1.05rem;font-weight:700">{ui.e(sm.get('headline', ''))}</div></div>
                  <div class="fs-muted">{ui.e(sm.get('tone', ''))}</div>
                  <div><div class="fs-side-h">New risks this year</div>{risks or '<span class="fs-muted">None found</span>'}</div>
                  <div><div class="fs-side-h">Removed or softened</div>{gone or '<span class="fs-muted">None found</span>'}</div>
                </div>""")
            sec_name = st.segmented_control("Section", list(res["sections"]), default="Risk Factors", key="chg_section")
            sec = res["sections"].get(sec_name or "Risk Factors", {})
            t_new, t_edit, t_del = st.tabs([f"New ({len(sec.get('added', []))})", f"Edited ({len(sec.get('edited', []))})",
                                            f"Removed ({len(sec.get('removed', []))})"])
            with t_new:
                html("".join(ui.para(fc.tag(ct, res["fy_new"], sec_name, "new", d["n"]), d["text"], "add")
                             for d in sec.get("added", [])) or ui.empty("No new paragraphs."))
            with t_edit:
                for d in sorted(sec.get("edited", []), key=lambda d: d["similarity"])[:25]:
                    html(f'<div class="fs-para"><span class="ptag">{ui.e(fc.tag(ct, res["fy_new"], sec_name, "edited", d["n"]))} '
                         f'· {d["similarity"]:.0%} same words</span>{ui.diff_html(fc.word_diff(d["old_text"], d["text"]))}</div>')
                if not sec.get("edited"):
                    html(ui.empty("No edited paragraphs."))
            with t_del:
                html("".join(ui.para(fc.tag(ct, res["fy_old"], sec_name, "removed", d["n"]), d["text"], "del")
                             for d in sec.get("removed", [])) or ui.empty("Nothing removed."))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
with tab_port:
    top_l, top_r = st.columns([4, 1], vertical_alignment="bottom")
    if top_r.button("Mark to market now", width="stretch"):
        out = broker.mark_to_market()
        exits = ", ".join(f"{t} ({r.replace('_', '-')}, P&L {money(pnl)})" for t, r, pnl in out["exits"]) or "none"
        st.toast(f"Equity {money(out['snapshot']['equity'], 2)} · automatic exits: {exits}")
        summ = broker.portfolio_summary()
    closed = pd.DataFrame(db.fetch_all("select realized_pnl from trades where realized_pnl is not null"))
    wins = f"{(closed['realized_pnl'] > 0).mean():.0%}" if len(closed) else "–"
    invested = 1 - summ["cash"] / summ["equity"] if summ["equity"] else 0
    pos_value = summ["equity"] - summ["cash"]
    html(ui.cards([
        {"label": "Portfolio value", "value": money(summ["equity"]), "sub": f"cash + stocks · started at {money(start_cash)}"},
        {"label": "Profit / loss", "value": signed_money(summ["equity"] - start_cash),
         "tone": ui.tone_of(summ["equity"] - start_cash), "sub": f"{pct((summ['equity'] / start_cash - 1) * 100, 2)} since start"},
        {"label": "Invested in stocks", "value": money(pos_value), "sub": f"{invested:.0%} of the portfolio · {len(summ['positions'])} positions"},
        {"label": "Cash", "value": money(summ["cash"]), "sub": f"{1 - invested:.0%} of the portfolio"},
        {"label": "Win rate (closed)", "value": wins, "sub": f"{len(closed)} closed trades"},
    ]))

    snaps = pd.DataFrame(db.fetch_all("select ts, cash, positions_value, equity, benchmark from snapshots order by ts"))
    g_l, g_r = st.columns([2.3, 1])
    with g_l:
        view = st.segmented_control("Chart", ["Value", f"Growth vs {BENCH}"], default="Value", key="port_view",
                                    label_visibility="collapsed")
        if len(snaps) > 1 and view == "Value":
            snaps["ts"] = pd.to_datetime(snaps["ts"]).dt.tz_localize("UTC").dt.tz_convert(MK["timezone"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=snaps["ts"], y=snaps["equity"], name="Portfolio value", mode="lines+markers",
                line=dict(color=C["accent"], width=2.4), marker=dict(size=5),
                customdata=snaps[["positions_value", "cash"]].values,
                hovertemplate=(f"%{{x|%d %b %H:%M}}<br>Value {CUR}%{{y:,.0f}}<br>Invested {CUR}%{{customdata[0]:,.0f}}"
                               f"<br>Cash {CUR}%{{customdata[1]:,.0f}}<extra></extra>")))
            fig.add_hline(y=start_cash, line=dict(color=C["muted"], dash="dot", width=1),
                          annotation_text=f"starting value {money(start_cash)}", annotation_position="bottom left")
            fig.update_layout(height=420, title=f"Portfolio value · {MK['currency']}", yaxis=dict(tickprefix=CUR, tickformat=",.0f"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        elif len(snaps) > 1 and snaps["benchmark"].notna().any():
            snaps["ts"] = pd.to_datetime(snaps["ts"]).dt.tz_localize("UTC").dt.tz_convert(MK["timezone"])
            b0 = snaps["benchmark"].dropna().iloc[0]
            snaps["port"] = snaps["equity"] / start_cash * 100
            snaps["bench"] = snaps["benchmark"] / b0 * 100
            snaps["dd"] = (snaps["equity"] / snaps["equity"].cummax() - 1) * 100
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=snaps["ts"], y=snaps["port"], name="Portfolio", line=dict(color=C["accent"], width=2.2),
                                     fill="tozeroy", fillcolor="rgba(122,162,255,.08)"), 1, 1)
            fig.add_trace(go.Scatter(x=snaps["ts"], y=snaps["bench"], name=BENCH, line=dict(color=C["muted"], width=1.6, dash="dot")), 1, 1)
            fig.add_trace(go.Scatter(x=snaps["ts"], y=snaps["dd"], name="Drawdown", line=dict(color=C["down"], width=1),
                                     fill="tozeroy", fillcolor="rgba(255,107,129,.18)"), 2, 1)
            lo = min(snaps["port"].min(), snaps["bench"].min())
            hi = max(snaps["port"].max(), snaps["bench"].max())
            fig.update_yaxes(range=[lo - 1, hi + 1], row=1, col=1)
            fig.update_yaxes(ticksuffix="%", row=2, col=1)
            fig.update_layout(height=420, title=f"Growth of 100 · portfolio vs {BENCH}")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            html(ui.empty("The growth chart appears after the first two portfolio snapshots "
                          "(the scheduler takes one every 15 minutes in market hours, or press Mark to market)."))
    with g_r:
        html(ui.section("Risk exposure", "against the fund's hard limits"))
        largest = max((p["weight_pct"] for p in summ["positions"]), default=0)
        sect = summ["sector_weights_pct"]
        top_sector = max(sect.items(), key=lambda kv: kv[1]) if sect else ("–", 0)
        html('<div class="fs-card" style="gap:14px">'
             + ui.meter("Capital invested", invested * 100, None, "rest is cash")
             + ui.meter("Largest position", largest, config.MAX_POSITION_PCT * 100)
             + ui.meter(f"Largest sector ({top_sector[0]})", top_sector[1], config.MAX_SECTOR_PCT * 100)
             + "</div>")

    pos = pd.DataFrame(db.fetch_all("select * from positions"))
    html(ui.section("Open positions", f"stops and targets are watched every 15 minutes in market hours"))
    if not pos.empty:
        pos["value"] = pos["shares"] * pos["last_price"]
        pos["pnl"] = (pos["last_price"] - pos["avg_price"]) * pos["shares"]
        pos["pnl_pct"] = (pos["last_price"] / pos["avg_price"] - 1) * 100
        pos["weight"] = pos["value"] / summ["equity"] * 100
        p_l, p_r = st.columns([2.3, 1])
        p_l.dataframe(pos[["ticker", "shares", "avg_price", "last_price", "pnl", "pnl_pct", "weight", "stop_loss", "take_profit", "sector"]],
                      hide_index=True, width="stretch", column_config={
                          "ticker": "Stock", "avg_price": st.column_config.NumberColumn("Avg cost", format=f"{CUR}%.2f"),
                          "last_price": st.column_config.NumberColumn("Last", format=f"{CUR}%.2f"),
                          "pnl": st.column_config.NumberColumn("P&L", format=f"{CUR}%.0f"),
                          "pnl_pct": st.column_config.NumberColumn("P&L %", format="%+.1f%%"),
                          "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=config.MAX_POSITION_PCT * 100, format="%.1f%%"),
                          "stop_loss": st.column_config.NumberColumn("Stop", format=f"{CUR}%.2f"),
                          "take_profit": st.column_config.NumberColumn("Target", format=f"{CUR}%.2f")})
        alloc = pd.concat([pos[["ticker", "value", "sector"]],
                           pd.DataFrame([{"ticker": "Cash", "value": summ["cash"], "sector": "Cash"}])])
        fig = px.sunburst(alloc, path=["sector", "ticker"], values="value", title="Allocation by sector")
        fig.update_layout(height=330, margin=dict(l=4, r=4, t=36, b=4))
        p_r.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        html(ui.empty("No open positions yet. Run the agents on a stock from the Trade desk."))

    a_l, a_r = st.columns([1, 1])
    with a_l:
        html(ui.section("Automatic activity", f"times in {TZ_LABEL}"))
        act = recent_activity(14)
        html(ui.timeline([(ACTIVITY_ICON[a["kind"]], a["when"], a["detail"]) for a in act]) if act
             else ui.empty("Automatic sales, queued orders and daily agent runs will appear here."))
    with a_r:
        pend = pd.DataFrame(db.fetch_all(
            "select ts, ticker, action, confidence, price as decided_at_price from signals where status = 'pending' order by ts"))
        html(ui.section("Orders waiting for the open", f"filled automatically when {market_clock.EXCHANGE} opens"))
        if not pend.empty:
            st.dataframe(local_time(pend, "ts"), hide_index=True, width="stretch")
        else:
            html(ui.empty("No queued orders."))

    html(ui.section("Decisions", f"latest 50 · times in {TZ_LABEL}"))
    sigs = pd.DataFrame(db.fetch_all(
        "select ts, ticker, action, confidence, price, stop_loss, shares, status, rationale from signals order by ts desc limit 50"))
    if sigs.empty:
        html(ui.empty("No decisions yet."))
    else:
        sigs["status"] = sigs["status"].map(lambda v: STATUS_WORDS.get(v, v))
        st.dataframe(local_time(sigs, "ts"), hide_index=True, width="stretch", column_config={
            "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "rationale": st.column_config.TextColumn(width="large")})
    html(ui.section("Trades", f"latest 50 · times in {TZ_LABEL}"))
    trades_df = pd.DataFrame(db.fetch_all(
        "select ts, ticker, side, shares, price, reason, realized_pnl from trades order by ts desc limit 50"))
    if trades_df.empty:
        html(ui.empty("No trades yet. Every decision so far was HOLD or was blocked by the risk rules."))
    else:
        st.dataframe(local_time(trades_df, "ts"), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
with tab_rag:
    html(ui.section(f"Ask the {DOC}", f"answers use only the company's latest {MK['doc_name']}, with a citation after every claim"))
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
    examples = ["How does the company describe competition?", "What drives revenue growth?", "What are the supply-chain risks?"]
    ex_pick = c2.pills("Try", examples, label_visibility="collapsed")
    q = ex_pick or q
    if c2.button("Ask", type="primary") and q:
        with st.spinner("Retrieving and answering…"):
            out = rag.answer(rt, q)
        with st.container(border=True):
            html(f'<div class="fs-side-h">Answer · {ui.e(short(rt))}</div>')
            st.markdown(out["answer"])
        html(ui.section("Retrieved passages", "blend of meaning similarity and keyword overlap"))
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
@st.cache_data(ttl=3600, show_spinner="Backtesting the watchlist…")
def _backtest(tickers):
    from finsight import backtest
    r = backtest.run(list(tickers))
    return r["summary"], r["curves"], r["trades"]


with tab_bt:
    html(ui.section("Backtest lab", "momentum rules + the fund's risk engine on 5 years of prices (no AI, to avoid look-ahead bias)"))
    extra = (["AMZN", "GOOGL", "META", "KO", "PG", "CVX"] if config.MARKET == "US"
             else ["TCS.NS", "HINDUNILVR.NS", "MARUTI.NS", "LT.NS", "ULTRACEMCO.NS", "ONGC.NS"])
    b1, b2 = st.columns([4, 1], vertical_alignment="bottom")
    bt_t = b1.multiselect("Universe", config.WATCHLIST + extra, default=config.WATCHLIST)
    if bt_t and b2.button("Run backtest", type="primary", width="stretch"):
        st.session_state["bt"] = _backtest(tuple(bt_t))
    if "bt" in st.session_state:
        bsum, curves, trs = st.session_state["bt"]
        s, b = bsum["strategy"], bsum["benchmark_buy_hold"]
        html(ui.cards([
            {"label": "Strategy return", "value": pct(s["total_return_pct"]), "tone": ui.tone_of(s["total_return_pct"]),
             "sub": f"CAGR {pct(s['cagr_pct'])} · {BENCH} {pct(b['total_return_pct'])}"},
            {"label": "Max drawdown", "value": pct(s["max_drawdown_pct"], 1, False), "tone": "fs-down",
             "sub": f"{BENCH} {pct(b['max_drawdown_pct'], 1, False)}"},
            {"label": "Sharpe ratio", "value": s["sharpe"], "sub": f"{BENCH} {b['sharpe']}"},
            {"label": "Trades · win rate", "value": f"{bsum['n_trades']} · {bsum['win_rate_pct']}%",
             "sub": f"avg win {pct(bsum['avg_win_pct'])} · avg loss {pct(bsum['avg_loss_pct'])}"},
            {"label": "Capital invested", "value": f"{bsum['avg_capital_invested_pct']}%", "sub": "average over the test"},
        ]))
        l, r = st.columns([2.3, 1])
        cv = curves.copy()
        dd = (cv["Strategy"] / cv["Strategy"].cummax() - 1) * 100
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
        for col, colour, dash in zip(cv.columns, [C["accent"], C["muted"], C["accent2"]], [None, "dot", "dash"]):
            fig.add_trace(go.Scatter(x=cv.index, y=cv[col], name=col, line=dict(color=colour, width=2 if col == "Strategy" else 1.4, dash=dash)), 1, 1)
        fig.add_trace(go.Scatter(x=cv.index, y=dd, name="Strategy drawdown", line=dict(color=C["down"], width=1),
                                 fill="tozeroy", fillcolor="rgba(255,107,129,.18)"), 2, 1)
        fig.update_yaxes(ticksuffix="%", row=2, col=1)
        fig.update_layout(height=440, title=f"Equity curves · {bsum['period']}")
        l.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        exits = pd.DataFrame(list(bsum["exit_reasons"].items()), columns=["reason", "count"])
        fig = px.pie(exits, names="reason", values="count", hole=0.62, title="Why trades closed",
                     color="reason", color_discrete_map={"stop_loss": C["down"], "take_profit": C["up"], "trend_break": C["warn"]})
        fig.update_layout(height=230)
        r.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if len(trs):
            fig = px.histogram(trs, x="return_pct", nbins=30, title="Return per trade (%)")
            fig.update_traces(marker_color=C["accent"])
            fig.update_layout(height=200, bargap=0.05, showlegend=False)
            r.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with st.expander("Trade list"):
            st.dataframe(trs, hide_index=True, width="stretch")

    html(ui.section("Report search accuracy", "does the right passage come back for a question?"))
    ev_path = Path(__file__).resolve().parent.parent / "eval" / (
        "rag_eval_results.json" if config.MARKET == "US" else f"rag_eval_results_{config.MARKET.lower()}.json")
    if ev_path.exists():
        ev_res = json.loads(ev_path.read_text())
        st.caption(f"{ev_res['n_questions']} paraphrased questions over {', '.join(ev_res.get('tickers', ['AAPL', 'NVDA', 'JNJ']))} "
                   f"{DOC}s. A hit means a retrieved passage contains the ground-truth fact. "
                   "Run `python eval/run_rag_eval.py` to refresh.")
        mdf = pd.DataFrame([{"configuration": k, **v["metrics"]} for k, v in ev_res["results"].items()])
        fig = px.bar(mdf.melt(id_vars="configuration", var_name="metric"), x="metric", y="value", color="configuration",
                     barmode="group", title="Meaning-only vs hybrid re-ranking")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.dataframe(mdf, hide_index=True, width="stretch")
    else:
        html(ui.empty("Run `python eval/run_rag_eval.py` to generate evaluation results."))


# ---------------------------------------------------------------------------
# Agent ops
# ---------------------------------------------------------------------------
with tab_ops:
    runs = local_time(pd.DataFrame(db.fetch_all(
        "select id, ticker, started_at, model, status, attempts, duration_s from runs order by id desc limit 50")), "started_at")
    ev = local_time(pd.DataFrame(db.fetch_all(
        "select ts, run_id, kind, name, duration_ms, detail from events order by id desc limit 500")), "ts")
    done = runs[runs["status"] != "running"] if len(runs) else runs
    recov = int(ev["kind"].isin(["llm_fallback", "data_fallback", "guardrail_retry", "risk_override", "risk_veto"]).sum()) if len(ev) else 0
    html(ui.cards([
        {"label": "Agent runs", "value": len(runs)},
        {"label": "Success rate", "value": f"{(done['status'] == 'success').mean():.0%}" if len(done) else "–"},
        {"label": "Average run time", "value": f"{runs['duration_s'].mean():.0f}s" if len(runs) and runs["duration_s"].notna().any() else "–"},
        {"label": "Tool calls", "value": int((ev["kind"] == "tool_call").sum()) if len(ev) else 0},
        {"label": "Safety catches", "value": recov, "sub": "fallbacks, guardrail retries, vetoes, overrides"},
    ]))
    if len(ev):
        l, r = st.columns(2)
        tc = ev[ev["kind"] == "tool_call"].groupby("name")["duration_ms"].agg(["count", "mean"]).reset_index()
        fig = px.bar(tc.sort_values("mean"), y="name", x="mean", orientation="h", title="Average tool latency (ms)",
                     text=tc.sort_values("mean")["count"].map(lambda n: f"{n} calls"))
        fig.update_traces(marker_color=C["accent"], textposition="outside")
        fig.update_layout(height=320, yaxis_title=None, xaxis_title=None)
        l.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        kinds = ev["kind"].value_counts().reset_index()
        kinds.columns = ["kind", "count"]
        fig = px.pie(kinds, names="kind", values="count", hole=0.6, title="Events by type")
        fig.update_layout(height=320)
        r.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        html(ui.section("Event log", f"tool calls, fallbacks, guardrails, vetoes · times in {TZ_LABEL}"))
        st.dataframe(ev, hide_index=True, width="stretch", height=320)
    else:
        html(ui.empty("Run the agents once and every tool call, retry and veto will be logged here."))
    html(ui.section("Runs", f"times in {TZ_LABEL}"))
    if len(runs):
        st.dataframe(runs, hide_index=True, width="stretch")
        rid = st.selectbox("Open the full report for run", runs["id"])
        rep = db.fetch_all("select report from runs where id = :i", i=int(rid))
        if rep and rep[0]["report"]:
            with st.expander("Full report", expanded=True):
                st.markdown(rep[0]["report"])
    else:
        html(ui.empty("No runs yet."))
