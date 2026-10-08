"""Draw docs/code_map.png + .svg: every Python file, what it does, and what flows between them.

Needs Graphviz (`brew install graphviz`).

    python docs/make_code_map.py
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent

# layer -> (fill colour, border colour, [(id, file, line 1, line 2), ...])
LAYERS = {
    "1. Entry points: you start these": ("#EEF2F7", "#7A8BA6", [
        ("streamlit_app", "app/streamlit_app.py", "The trading-desk website, 7 tabs", "Run button starts an analysis"),
        ("scheduler", "finsight/scheduler.py", "Autopilot: price checks every 15 min", "agents on all stocks after close"),
        ("run_crew", "scripts/run_crew.py", "Analyse a stock from the terminal", "--dry-run = no trade"),
        ("run_backtest", "scripts/run_backtest.py", "Replay 4 years of prices", "saves results to eval/"),
        ("fetch_reports", "scripts/fetch_annual_reports.py", "Download the 6 Indian reports", "and verify each PDF"),
        ("run_rag_eval", "eval/run_rag_eval.py", "Test report-search accuracy", "Hit@1/3/5 and MRR"),
        ("reset_portfolio", "scripts/reset_portfolio.py", "Clean slate before a demo", "deletes trades and positions"),
    ]),
    "2. Screen look": ("#F4EEFB", "#9B7FD0", [
        ("ui", "app/ui.py", "Colours, fonts, chart theme", "header, tape, cards, gauge, diff view"),
    ]),
    "3. Orchestration": ("#FDF3E3", "#C98A2B", [
        ("pipeline", "finsight/pipeline.py", "Runs ONE analysis end to end", "data, crew, policy, risk, save"),
        ("backtest", "finsight/backtest.py", "Momentum rules on past prices", "no AI (avoids look-ahead bias)"),
    ]),
    "4. Agents and AI": ("#EFEDFE", "#6E62D0", [
        ("crew", "finsight/crew.py", "3 agents + their tasks, fact sheet", "guardrail checks the decision"),
        ("crew_tools", "finsight/tools/crew_tools.py", "10 tools the agents can call", "cached, logged, never crash"),
        ("filing_changes", "finsight/filing_changes.py", "Agent 4: this year's 10-K vs last", "new / edited / removed risks"),
        ("llm", "finsight/llm.py", "Picks the AI model", "gpt-oss-120b, 20b, Qwen fallback"),
    ]),
    "5. Trading and time": ("#E5F5EF", "#2F9C78", [
        ("broker", "finsight/broker.py", "Risk engine + paper broker", "size, fill or queue, stops, snapshots"),
        ("market_clock", "finsight/market_clock.py", "NYSE / NSE opening hours", "open now? next open?"),
    ]),
    "6. Report library (RAG)": ("#E8F3E0", "#5E9A2E", [
        ("rag", "finsight/rag.py", "Cuts reports into passages, vectors", "search + Ask the Report answers"),
        ("pdf_reports", "finsight/pdf_reports.py", "Reads Indian PDFs", "keeps narrative pages, labels them"),
        ("report_bot", "finsight/report_bot.py", "Downloads Indian reports politely", "checks robots.txt and FY2026"),
    ]),
    "7. Data and valuation": ("#E6F0FB", "#3A7BC8", [
        ("market_data", "finsight/tools/market_data.py", "Prices and company numbers", "Yahoo, SEC backup, saved copies"),
        ("valuation", "finsight/tools/valuation.py", "All the maths: RSI, ATR, DCF", "reverse DCF, 0-100 quant score"),
    ]),
    "8. Foundation: used by almost every file": ("#F1F0EC", "#8A887F", [
        ("config", "finsight/config.py", "Reads .env: market, models", "database, risk rules"),
        ("markets", "finsight/markets.py", "US vs India settings", "currency, rates, hours, stocks"),
        ("db", "finsight/db.py", "The memory: 6 tables", "runs, signals, trades, positions, ..."),
    ]),
}

EXTERNAL = [
    ("yahoo", "Yahoo Finance"), ("sec", "SEC EDGAR"), ("groq", "Groq cloud AI"), ("ollama", "Ollama on the Mac"),
    ("chroma", "ChromaDB (data/chroma)"), ("postgres", "PostgreSQL (Docker)"), ("sites", "Company websites"),
]
ROWS = [  # one row per level, top to bottom; arrows mostly point down
    ["browser", "streamlit_app", "scheduler", "run_crew", "run_backtest", "fetch_reports", "run_rag_eval", "reset_portfolio"],
    ["ui", "pipeline", "backtest"],
    ["crew", "crew_tools", "filing_changes", "llm"],
    ["broker", "market_clock", "rag"],
    ["market_data", "valuation", "pdf_reports", "report_bot"],
    ["db", "config", "markets"],
    ["yahoo", "sec", "groq", "ollama", "chroma", "postgres", "sites"],
]

# (from, to, what flows)
EDGES = [
    ("streamlit_app", "pipeline", "Run button: analyse(stock)"),
    ("streamlit_app", "ui", "look + components"),
    ("streamlit_app", "broker", "portfolio, mark to market"),
    ("streamlit_app", "market_data", "prices for charts, tape"),
    ("streamlit_app", "valuation", "scores, DCF for the desk"),
    ("streamlit_app", "rag", "Ask the Report"),
    ("streamlit_app", "filing_changes", "What changed tab"),
    ("streamlit_app", "backtest", "Backtest lab tab"),
    ("streamlit_app", "market_clock", "open / closed badge"),
    ("scheduler", "pipeline", "daily agent run"),
    ("scheduler", "broker", "every 15 min: fills, stops, snapshot"),
    ("scheduler", "rag", "index reports at start"),
    ("scheduler", "filing_changes", "refresh 10-K comparisons"),
    ("scheduler", "market_clock", "is the market open?"),
    ("run_crew", "pipeline", "analyse from terminal"),
    ("run_backtest", "backtest", "run + save results"),
    ("fetch_reports", "report_bot", "download + verify"),
    ("run_rag_eval", "rag", "accuracy test questions"),
    ("reset_portfolio", "db", "delete trades, positions"),
    ("pipeline", "crew", "build + run the agents"),
    ("pipeline", "llm", "model chain, fallback"),
    ("pipeline", "crew_tools", "fetch data once (shared cache)"),
    ("pipeline", "broker", "plan_trade + execute"),
    ("pipeline", "rag", "index report if missing"),
    ("pipeline", "db", "save run + decision"),
    ("crew", "crew_tools", "agents call tools"),
    ("crew_tools", "market_data", "prices, fundamentals"),
    ("crew_tools", "valuation", "DCF, quant score"),
    ("crew_tools", "rag", "search_annual_report"),
    ("crew_tools", "filing_changes", "get_filing_changes"),
    ("crew_tools", "broker", "portfolio state, plan_position"),
    ("filing_changes", "rag", "10-K sections, vectors"),
    ("filing_changes", "llm", "model for agent 4"),
    ("filing_changes", "crew", "JSON + citation helpers"),
    ("rag", "pdf_reports", "useful PDF pages"),
    ("rag", "report_bot", "fetch a missing PDF"),
    ("rag", "llm", "model for answers"),
    ("broker", "market_data", "live prices, ATR"),
    ("broker", "valuation", "ATR for the stop-loss"),
    ("broker", "market_clock", "fill now or queue?"),
    ("broker", "db", "trades, positions, snapshots"),
    ("backtest", "market_data", "4 years of prices"),
    ("config", "markets", "per-market settings"),
    ("db", "config", "database address"),
]
OUTSIDE = [
    ("browser", "streamlit_app", "localhost:8502 / 8503"),
    ("market_data", "yahoo", "prices, fundamentals"),
    ("market_data", "sec", "backup numbers"),
    ("rag", "sec", "10-K text"),
    ("filing_changes", "sec", "last two 10-Ks"),
    ("rag", "chroma", "store + search passages"),
    ("rag", "ollama", "text to vectors"),
    ("llm", "groq", "agents' thinking"),
    ("llm", "ollama", "offline backup"),
    ("report_bot", "sites", "annual-report PDFs"),
    ("db", "postgres", "6 tables (SQLite backup)"),
]


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dot() -> str:
    out = ['digraph G {',
           'graph [rankdir=TB, newrank=true, splines=spline, nodesep=0.6, ranksep=1.5, fontname="Helvetica", fontsize=22, '
           'labelloc=t, label=<<b>FinSight Crew v2: every Python file and what flows between them</b><br/>'
           '<font point-size="13">Arrows point from the file that calls to the file it uses. '
           'Dashed lines go to outside services. tests/test_core.py (41 checks) and docs/ scripts are not drawn.</font>>, '
           'pad=0.4, bgcolor="white"];',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.15,0.08", penwidth=1.3];',
           'edge [fontname="Helvetica", fontsize=10, color="#7D8590", fontcolor="#2F3742", arrowsize=0.7, penwidth=1.1, decorate=true, labelfloat=false];']
    for i, (layer, (fill, border, nodes)) in enumerate(LAYERS.items()):
        out.append(f'subgraph cluster_{i} {{ label=<<b>{esc(layer)}</b>>; fontsize=13; style="rounded,dashed"; '
                   f'color="{border}"; fontcolor="{border}"; margin=14;')
        for nid, path, l1, l2 in nodes:
            out.append(f'{nid} [fillcolor="{fill}", color="{border}", label=<<b>{esc(path)}</b><br/>'
                       f'<font point-size="10" color="#3B4250">{esc(l1)}<br/>{esc(l2)}</font>>];')
        out.append('}')
    out.append('subgraph cluster_ext { label=<<b>Outside services</b>>; fontsize=13; style="rounded,dashed"; '
               'color="#B0B4BA"; fontcolor="#7D8590"; margin=14;')
    for nid, name in EXTERNAL:
        out.append(f'{nid} [shape=box, style="rounded,dashed", color="#9AA0A8", fillcolor="white", fontsize=11, '
                   f'label=<<b>{esc(name)}</b>>];')
    out.append('}')
    out.append('browser [shape=box, style="rounded,dashed", color="#9AA0A8", fillcolor="white", fontsize=11, '
               'label=<<b>Your browser</b>>];')
    for row in ROWS:
        out.append("{rank=same; " + "; ".join(row) + ";}")
    for upper, lower in zip(ROWS, ROWS[1:]):
        out.append(f"{upper[1]} -> {lower[1]} [style=invis, weight=20];")
    for a, b, lbl in EDGES:
        out.append(f'{a} -> {b} [label=<{esc(lbl)}>];')
    for a, b, lbl in OUTSIDE:
        out.append(f'{a} -> {b} [label=<{esc(lbl)}>, style=dashed, color="#A8AEB6"];')
    out.append('}')
    return "\n".join(out)


if __name__ == "__main__":
    src = OUT / "code_map.dot"
    src.write_text(dot())
    for fmt, extra in (("svg", []), ("png", ["-Gdpi=150"])):
        subprocess.run(["dot", f"-T{fmt}", *extra, str(src), "-o", str(OUT / f"code_map.{fmt}")], check=True)
    print("wrote", OUT / "code_map.png", "and", OUT / "code_map.svg")
