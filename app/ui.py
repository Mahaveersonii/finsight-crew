"""Visual layer for the Streamlit app: theme, Plotly template and small HTML components.

Design: a dark trading-terminal look. Numbers in a monospaced face so columns line up,
green / coral only for up / down, one periwinkle accent for interactive things.
"""
from html import escape

import plotly.graph_objects as go
import plotly.io as pio

C = {
    "bg": "#0B1020", "panel": "#111830", "panel2": "#172040", "line": "#24304F",
    "text": "#E8EDF8", "muted": "#8C98B4", "accent": "#7AA2FF", "accent2": "#B892FF",
    "up": "#3DDC97", "down": "#FF6B81", "warn": "#F7B955",
}
FONT = "Manrope, system-ui, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {{
  --bg:{C['bg']}; --panel:{C['panel']}; --panel2:{C['panel2']}; --line:{C['line']};
  --text:{C['text']}; --muted:{C['muted']}; --accent:{C['accent']}; --accent2:{C['accent2']};
  --up:{C['up']}; --down:{C['down']}; --warn:{C['warn']};
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {{ font-family:{FONT}; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, #1A2550 0%, transparent 60%), var(--bg); }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stSidebar"] {{ background: #0D1328; border-right: 1px solid var(--line); }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}

/* tabs as a segmented bar */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: var(--panel); padding: 5px; border-radius: 12px; border: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ height: 38px; padding: 0 16px; border-radius: 8px; color: var(--muted); font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: var(--panel2); color: var(--text); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* streamlit widgets */
[data-testid="stDataFrame"], [data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
.stButton > button[kind="primary"] {{ background: linear-gradient(135deg, var(--accent), var(--accent2)); border: 0; color: #0B1020; font-weight: 700; }}
.stButton > button {{ border-radius: 10px; }}
[data-testid="stPlotlyChart"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 6px 6px 0; }}

/* components */
.fs-num {{ font-family:{MONO}; font-variant-numeric: tabular-nums; }}
.fs-hl {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px; min-width:0; }}
.fs-hr {{ display:flex; gap:26px; align-items:center; }}
.fs-header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:14px 22px; padding: 14px 18px; background: var(--panel);
  border:1px solid var(--line); border-radius:16px; margin-bottom: 10px; }}
.fs-brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; }}
.fs-logo {{ width:30px; height:30px; border-radius:9px; background: linear-gradient(135deg, var(--accent), var(--accent2));
  display:grid; place-items:center; color:#0B1020; font-weight:800; font-size:.9rem; }}
.fs-brand small {{ display:block; font-weight:500; color:var(--muted); font-size:.72rem; letter-spacing:.02em; }}
.fs-spacer {{ flex:1; }}
.fs-hstat {{ display:grid; gap:1px; }}
.fs-hstat span {{ font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
.fs-hstat b {{ font-family:{MONO}; font-size:1rem; font-weight:600; }}

.fs-pill {{ display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:999px; font-size:.78rem; font-weight:600;
  border:1px solid var(--line); background: var(--panel2); color: var(--text); white-space:nowrap; }}
.fs-dot {{ width:7px; height:7px; border-radius:50%; background: var(--muted); }}
.fs-up {{ color: var(--up); }} .fs-down {{ color: var(--down); }} .fs-warn {{ color: var(--warn); }} .fs-acc {{ color: var(--accent); }} .fs-muted {{ color: var(--muted); }}
.fs-pill.up {{ border-color: rgba(61,220,151,.35); background: rgba(61,220,151,.08); }} .fs-pill.up .fs-dot {{ background: var(--up); box-shadow:0 0 0 4px rgba(61,220,151,.15); }}
.fs-pill.down {{ border-color: rgba(255,107,129,.35); background: rgba(255,107,129,.08); }} .fs-pill.down .fs-dot {{ background: var(--down); }}
.fs-pill.warn {{ border-color: rgba(247,185,85,.35); background: rgba(247,185,85,.08); }} .fs-pill.warn .fs-dot {{ background: var(--warn); }}
.fs-pill.acc {{ border-color: rgba(122,162,255,.35); background: rgba(122,162,255,.08); }} .fs-pill.acc .fs-dot {{ background: var(--accent); }}

.fs-tape {{ overflow:hidden; border:1px solid var(--line); border-radius:12px; background: var(--panel); margin-bottom: 14px; }}
.fs-tape-track {{ display:flex; gap:34px; width:max-content; padding:9px 18px; animation: fs-scroll 40s linear infinite; }}
.fs-tape:hover .fs-tape-track {{ animation-play-state: paused; }}
.fs-tick {{ display:flex; gap:9px; align-items:baseline; font-size:.85rem; white-space:nowrap; }}
.fs-tick b {{ font-weight:700; }}
@keyframes fs-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .fs-tape-track {{ animation: none; }} }}

.fs-grid {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin: 4px 0 14px; }}
.fs-card {{ background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; display:grid; gap:4px; }}
.fs-card .lbl {{ font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
.fs-card .val {{ font-family:{MONO}; font-size:1.45rem; font-weight:600; line-height:1.15; }}
.fs-card .sub {{ font-size:.8rem; color:var(--muted); }}
.fs-section {{ display:flex; align-items:baseline; gap:10px; margin: 18px 0 8px; }}
.fs-section .t {{ font-size:1.08rem; font-weight:700; color: var(--text); }}
.fs-section span {{ color: var(--muted); font-size:.8rem; }}

.fs-steps {{ display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:8px; margin: 6px 0 10px; }}
.fs-step {{ border:1px solid var(--line); background: var(--panel); border-radius:12px; padding:10px 12px; display:grid; gap:2px; }}
.fs-step b {{ font-size:.85rem; }} .fs-step span {{ font-size:.72rem; color: var(--muted); }}
.fs-step.done {{ border-color: rgba(61,220,151,.4); }} .fs-step.done b::before {{ content:"✓ "; color: var(--up); }}
.fs-step.live {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(122,162,255,.15); }}
.fs-step.live b::before {{ content:"● "; color: var(--accent); }}
@media (max-width: 900px) {{ .fs-steps {{ grid-template-columns: repeat(3, minmax(0,1fr)); }} }}

.fs-term {{ background:#070B16; border:1px solid var(--line); border-radius:12px; padding:12px 14px; font-family:{MONO}; font-size:.8rem;
  line-height:1.55; color:#C9D3EA; max-height: 300px; overflow:auto; white-space: pre-wrap; }}

.fs-decision {{ background: var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; display:grid; gap:14px; }}
.fs-action {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
.fs-action .big {{ font-size:1.6rem; font-weight:800; padding:6px 16px; border-radius:12px; letter-spacing:.04em; }}
.big.BUY {{ background: rgba(61,220,151,.14); color: var(--up); }} .big.SELL {{ background: rgba(255,107,129,.14); color: var(--down); }}
.big.HOLD {{ background: rgba(247,185,85,.14); color: var(--warn); }}

.fs-rr {{ position:relative; height:36px; margin: 22px 4px 26px; }}
.fs-rr .bar {{ position:absolute; top:14px; height:8px; border-radius:4px; }}
.fs-rr .mk {{ position:absolute; top:4px; width:2px; height:28px; background: var(--text); }}
.fs-rr .tag {{ position:absolute; font-family:{MONO}; font-size:.72rem; white-space:nowrap; transform: translateX(-50%); }}
.fs-rr .tag.t {{ top:-16px; }} .fs-rr .tag.b {{ top:36px; color: var(--muted); }}

.fs-chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.fs-chip {{ font-size:.76rem; padding:3px 9px; border-radius:7px; border:1px solid var(--line); background: var(--panel2); }}
.fs-chip.cite {{ font-family:{MONO}; font-size:.72rem; color: var(--accent); }}

.fs-meter {{ display:grid; gap:6px; }}
.fs-meter .top {{ display:flex; justify-content:space-between; font-size:.82rem; }}
.fs-meter .track {{ position:relative; height:8px; border-radius:4px; background: var(--panel2); overflow:hidden; }}
.fs-meter .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:4px; }}
.fs-meter .lim {{ position:absolute; top:-3px; width:2px; height:14px; background: var(--muted); }}

.fs-tl {{ display:grid; gap:0; }}
.fs-tl-row {{ display:grid; grid-template-columns: 26px 92px 1fr; gap:10px; padding:9px 4px; border-bottom:1px solid var(--line); font-size:.86rem; }}
.fs-tl-row:last-child {{ border-bottom:0; }}
.fs-tl-row .when {{ font-family:{MONO}; color: var(--muted); font-size:.76rem; padding-top:2px; }}
.fs-diff {{ font-size:.9rem; line-height:1.65; }}
.fs-diff .add {{ background: rgba(61,220,151,.16); color: var(--up); border-radius:3px; padding:0 2px; }}
.fs-diff .del {{ background: rgba(255,107,129,.13); color: var(--down); text-decoration: line-through; border-radius:3px; padding:0 2px; }}
.fs-para {{ border-left:3px solid var(--line); padding:6px 0 6px 12px; margin:8px 0; font-size:.9rem; }}
.fs-para.add {{ border-color: var(--up); }} .fs-para.del {{ border-color: var(--down); color: var(--muted); }}
.fs-para .ptag {{ font-family:{MONO}; font-size:.72rem; color: var(--accent); display:block; margin-bottom:3px; }}
.fs-empty {{ border:1px dashed var(--line); border-radius:14px; padding:22px; text-align:center; color: var(--muted); }}
.fs-side-h {{ font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin: 14px 0 6px; }}
</style>
"""

pio.templates["finsight"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=C["text"], size=12),
    colorway=[C["accent"], C["up"], C["warn"], C["accent2"], C["down"], "#5FD3F3", "#9AA6C2"],
    xaxis=dict(gridcolor="rgba(140,152,180,.10)", zeroline=False, linecolor=C["line"]),
    yaxis=dict(gridcolor="rgba(140,152,180,.10)", zeroline=False, linecolor=C["line"]),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=36, b=8), hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"], font_family=MONO),
    title=dict(font=dict(size=14), x=0.01),
))
pio.templates.default = "plotly_dark+finsight"


def e(x) -> str:
    return escape(str(x))


def tone_of(x) -> str:
    return "" if x is None else ("fs-up" if x > 0 else "fs-down" if x < 0 else "fs-muted")


def pill(text, tone="", dot=True) -> str:
    return f'<span class="fs-pill {tone}">{"<i class=fs-dot></i>" if dot else ""}{e(text)}</span>'


def header(market_name, flag, mstatus, model, equity_txt, pnl_txt, pnl_val, clock_txt) -> str:
    mk = pill(f"{mstatus['label']} · {mstatus['detail']}", "up" if mstatus["open"] else "down")
    clock = pill(clock_txt, "", dot=False)
    ai = pill(f"Agents ready · {model}" if model else "No AI model reachable", "acc" if model else "warn")
    return (f'<div class="fs-header"><div class="fs-hl">'
            f'<div class="fs-brand"><div class="fs-logo">FS</div><div>FinSight Crew<small>Agentic research &amp; '
            f'paper-trading desk · {flag} {e(market_name)}</small></div></div>{mk}{ai}{clock}</div>'
            f'<div class="fs-hr"><div class="fs-hstat"><span>Equity</span><b>{e(equity_txt)}</b></div>'
            f'<div class="fs-hstat"><span>Since start</span><b class="{tone_of(pnl_val)}">{e(pnl_txt)}</b></div></div></div>')


def tape(items) -> str:
    """items: (ticker, price_text, pct). Duplicated so the CSS loop is seamless."""
    cells = "".join(
        f'<div class="fs-tick"><b>{e(t)}</b><span class="fs-num">{e(p)}</span>'
        f'<span class="fs-num {tone_of(c)}">{"▲" if (c or 0) >= 0 else "▼"} {abs(c or 0):.2f}%</span></div>'
        for t, p, c in items)
    return f'<div class="fs-tape" aria-label="Watchlist prices"><div class="fs-tape-track">{cells}{cells}</div></div>'


def cards(items) -> str:
    """items: dicts with label, value, optional sub, tone."""
    out = []
    for it in items:
        out.append(f'<div class="fs-card"><span class="lbl">{e(it["label"])}</span>'
                   f'<span class="val {it.get("tone", "")}">{e(it["value"])}</span>'
                   + (f'<span class="sub">{e(it["sub"])}</span>' if it.get("sub") else "") + "</div>")
    return f'<div class="fs-grid">{"".join(out)}</div>'


def section(title, note="") -> str:
    return f'<div class="fs-section"><span class="t">{e(title)}</span><span>{e(note)}</span></div>'


STEPS = [("Market data", "prices, fundamentals"), ("Report search", "RAG over filings"), ("Analyst", "valuation + evidence"),
         ("Portfolio manager", "decision + sizing"), ("Guardrails", "format, citations, facts"), ("Risk & broker", "limits, order")]


def stage_from_log(lines) -> int:
    """Map the pipeline's progress messages to the step reached (index into STEPS; 6 = finished)."""
    stage = 0
    for ln in lines:
        if "Indexing" in ln or "search_annual_report" in ln:
            stage = max(stage, 1)
        if any(k in ln for k in ("run_valuation", "get_past_decisions")):
            stage = max(stage, 2)
        if any(k in ln for k in ("get_portfolio_state", "plan_position")):
            stage = max(stage, 3)
        if "Guardrail" in ln:
            stage = max(stage, 4)
        if "Policy check" in ln or "Risk engine" in ln:
            stage = max(stage, 5)
        if ln.startswith("✅"):
            stage = 6
    return stage


def stepper(stage) -> str:
    cells = []
    for i, (name, note) in enumerate(STEPS):
        cls = "done" if i < stage else ("live" if i == stage else "")
        cells.append(f'<div class="fs-step {cls}"><b>{e(name)}</b><span>{e(note)}</span></div>')
    return f'<div class="fs-steps">{"".join(cells)}</div>'


def terminal(lines) -> str:
    return f'<div class="fs-term">{e(chr(10).join(lines))}</div>'


def rr_bar(stop, entry, target, fmt) -> str:
    """Stop - entry - target on one line, with the reward/risk multiple."""
    lo, hi = stop, target
    span = (hi - lo) or 1
    pos = (entry - lo) / span * 100
    r = (target - entry) / (entry - stop) if entry > stop else None
    return f"""<div class="fs-rr">
      <div class="bar" style="left:0;width:{pos:.1f}%;background:rgba(255,107,129,.55)"></div>
      <div class="bar" style="left:{pos:.1f}%;width:{100 - pos:.1f}%;background:rgba(61,220,151,.55)"></div>
      <div class="mk" style="left:{pos:.1f}%"></div>
      <span class="tag t fs-down" style="left:0%;transform:none">Stop {e(fmt(stop))}</span>
      <span class="tag t" style="left:{pos:.1f}%">Entry {e(fmt(entry))}</span>
      <span class="tag t fs-up" style="left:100%;transform:translateX(-100%)">Target {e(fmt(target))}</span>
      <span class="tag b" style="left:{pos / 2:.1f}%">risk</span>
      <span class="tag b" style="left:{pos + (100 - pos) / 2:.1f}%">reward{f" · {r:.1f}R" if r else ""}</span>
    </div>"""


def chips(items, cls="") -> str:
    return '<div class="fs-chips">' + "".join(f'<span class="fs-chip {cls}">{e(x)}</span>' for x in items) + "</div>"


def meter(label, value_pct, limit_pct=None, note="") -> str:
    v = max(0.0, min(100.0, value_pct or 0))
    over = limit_pct is not None and v > limit_pct
    colour = C["down"] if over else (C["warn"] if limit_pct and v > 0.8 * limit_pct else C["accent"])
    lim = f'<div class="lim" style="left:{limit_pct}%"></div>' if limit_pct is not None else ""
    return (f'<div class="fs-meter"><div class="top"><span>{e(label)}</span><span class="fs-num">{v:.1f}%'
            f'{f" / {limit_pct:g}% limit" if limit_pct is not None else ""}</span></div>'
            f'<div class="track"><div class="fill" style="width:{v}%;background:{colour}"></div>{lim}</div>'
            + (f'<span class="fs-muted" style="font-size:.74rem">{e(note)}</span>' if note else "") + "</div>")


def timeline(rows) -> str:
    """rows: (icon, when, html_safe_text)"""
    return '<div class="fs-card fs-tl">' + "".join(
        f'<div class="fs-tl-row"><span>{icon}</span><span class="when">{e(when)}</span><span>{e(text)}</span></div>'
        for icon, when, text in rows) + "</div>"


def empty(text) -> str:
    return f'<div class="fs-empty">{e(text)}</div>'


def gauge(value, threshold=70, title="Composite score"):
    v = value or 0
    colour = C["up"] if v >= threshold else (C["warn"] if v >= 50 else C["down"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"font": {"family": MONO, "size": 40}},
        title={"text": title, "font": {"size": 13, "color": C["muted"]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": C["muted"], "tickwidth": 1},
               "bar": {"color": colour, "thickness": 0.28}, "bgcolor": C["panel2"], "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "rgba(255,107,129,.08)"},
                         {"range": [50, threshold], "color": "rgba(247,185,85,.08)"},
                         {"range": [threshold, 100], "color": "rgba(61,220,151,.08)"}],
               "threshold": {"line": {"color": C["text"], "width": 2}, "thickness": 0.8, "value": threshold}}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=0))
    return fig


def diff_html(runs) -> str:
    return '<div class="fs-diff">' + " ".join(
        f'<span class="{op}">{e(t)}</span>' if op != "same" else e(t) for op, t in runs) + "</div>"


def para(tag_text, text, kind="") -> str:
    return f'<div class="fs-para {kind}"><span class="ptag">{e(tag_text)}</span>{e(text)}</div>'
