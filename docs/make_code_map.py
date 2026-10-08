"""Draw docs/code_map.svg + .png: every Python file, what it does, and the main flow between them.

Hand-placed grid (one row per layer, right-angle lines) so the picture stays readable.
Secondary links are listed inside the boxes ("uses: ...") instead of drawn.
PNG export uses headless Google Chrome.

    python docs/make_code_map.py
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
W, H = 1800, 1050
BW, BH = 240, 70                          # box width / height
COL = [170, 460, 750, 1040, 1330, 1620]   # box centres, x
ROW = [110, 270, 440, 620, 790]           # box tops, y

ROWS = [  # (label, fill, border)
    ("1. Entry points: you start these", "#EEF2F7", "#6F84A6"),
    ("2. Orchestration", "#FDF3E3", "#C98A2B"),
    ("3. Agents and AI", "#EFEDFE", "#6E62D0"),
    ("4. Services", "#E6F0FB", "#3A7BC8"),
    ("5. Helpers", "#E5F5EF", "#2F9C78"),
]

# id: (row, col, file, line, small line)
BOXES = {
    "run_crew": (0, 0, "scripts/run_crew.py", "Analyse a stock in the terminal", "--dry-run = no trade"),
    "streamlit_app": (0, 1, "app/streamlit_app.py", "Trading-desk website, 7 tabs", "uses ui.py + most services"),
    "scheduler": (0, 2, "finsight/scheduler.py", "Autopilot: 15-min + daily runs", "also: rag, filing_changes"),
    "run_rag_eval": (0, 3, "eval/run_rag_eval.py", "Tests report-search accuracy", "uses: rag"),
    "run_backtest": (0, 4, "scripts/run_backtest.py", "Replays 4 years of prices", "saves results to eval/"),
    "fetch_reports": (0, 5, "scripts/fetch_annual_reports.py", "Downloads 6 Indian reports", "and verifies each PDF"),
    "pipeline": (1, 1, "finsight/pipeline.py", "Runs ONE analysis end to end", "data, agents, policy, trade, save"),
    "backtest": (1, 4, "finsight/backtest.py", "Momentum rules, no AI", "uses: market_data"),
    "llm": (2, 0, "finsight/llm.py", "Picks the AI model", "gpt-oss-120b, 20b, Qwen"),
    "crew": (2, 1, "finsight/crew.py", "3 agents + fact sheet", "guardrail checks the decision"),
    "crew_tools": (2, 2, "finsight/tools/crew_tools.py", "10 tools the agents call", "cached, logged, never crash"),
    "filing_changes": (2, 3, "finsight/filing_changes.py", "Agent 4: 10-K vs last year", "uses: rag, llm"),
    "valuation": (3, 1, "finsight/tools/valuation.py", "RSI, ATR, DCF, quant score", "pure maths, no internet"),
    "market_data": (3, 2, "finsight/tools/market_data.py", "Prices + company numbers", "Yahoo, SEC backup, cache"),
    "rag": (3, 3, "finsight/rag.py", "Report library + search", "also answers Ask the Report"),
    "broker": (3, 4, "finsight/broker.py", "Risk engine + paper broker", "size, fill or queue, stops"),
    "market_clock": (3, 5, "finsight/market_clock.py", "NYSE / NSE opening hours", "open now? next open?"),
    "pdf_reports": (4, 2, "finsight/pdf_reports.py", "Reads Indian PDFs", "keeps narrative pages"),
    "report_bot": (4, 3, "finsight/report_bot.py", "Downloads Indian reports", "respects robots.txt"),
}

# (path, label, label x, label y, anchor)
EDGES = [
    ("M170 180 V228 H400 V268", "terminal run", 285, 220, "middle"),
    ("M460 180 V268", "Run button", 468, 214, "start"),
    ("M710 180 V228 H520 V268", "daily run", 615, 220, "middle"),
    ("M790 180 V196 H1185 V600 H1290 V618", "every 15 min", 1193, 470, "start"),
    ("M580 305 H1185", "size + trade (joins the line to broker)", 880, 297, "middle"),
    ("M1330 180 V268", "replay", 1338, 224, "start"),
    ("M460 340 V438", "run the agents", 468, 394, "start"),
    ("M400 340 V384 H170 V438", "choose model", 285, 376, "middle"),
    ("M580 475 H628", "calls tools", 604, 432, "middle"),
    ("M870 475 H918", "what changed", 894, 432, "middle"),
    ("M750 510 V618", "prices", 758, 590, "start"),
    ("M700 510 V560 H460 V618", "DCF, score", 580, 552, "middle"),
    ("M800 510 V560 H1040 V618", "report search", 920, 552, "middle"),
    ("M1450 655 H1498", "open?", 1474, 646, "middle"),
    ("M1000 690 V740 H750 V788", "PDF pages", 875, 732, "middle"),
    ("M1080 690 V788", "missing PDF", 1088, 744, "start"),
    ("M1740 145 H1768 V825 H1162", "download + verify", 1460, 817, "middle"),
]

FOUNDATION = ("6. Foundation, used by every file:   config.py reads .env (market, models, risk rules)   ·   "
              "markets.py holds US vs India settings   ·   db.py stores 6 tables (runs, signals, trades, positions, "
              "snapshots, events)")
OUTSIDE = ("7. Outside services:   Yahoo Finance ← market_data   ·   SEC EDGAR ← market_data, rag, filing_changes"
           "   ·   Groq AI ← llm   ·   Ollama + ChromaDB ← rag   ·   PostgreSQL ← db   ·   "
           "Company websites ← report_bot   ·   Browser → streamlit_app")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg() -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#5B6472"/></marker></defs>',
         '<text x="40" y="44" font-size="24" font-weight="700" fill="#1F2630">FinSight Crew v2: how the Python files connect</text>',
         '<text x="40" y="70" font-size="14" fill="#5B6472">Arrows show the main flow, from the file that calls to the file '
         'it uses. The small grey line in each box says what else it uses.</text>']
    for i, (label, fill, border) in enumerate(ROWS):
        y0 = ROW[i] - 26
        o.append(f'<rect x="30" y="{y0}" width="{W - 60}" height="{BH + 40}" rx="12" fill="#FAFBFC" stroke="{border}" '
                 f'stroke-width="1" stroke-dasharray="5 4"/>')
        o.append(f'<text x="44" y="{y0 + 17}" font-size="13" font-weight="700" fill="{border}">{esc(label)}</text>')
    for path, label, lx, ly, anchor in EDGES:
        joins = "joins the line" in label
        marker = "" if joins else ' marker-end="url(#a)"'
        o.append(f'<path d="{path}" fill="none" stroke="#5B6472" stroke-width="1.6"{marker}/>')
        if joins:  # a dot where it merges into the line to broker.py
            x_end = path.split("H")[-1]
            o.append(f'<circle cx="{x_end}" cy="305" r="4" fill="#5B6472"/>')
            label = label.replace(" (joins the line to broker)", "")
        o.append(f'<text x="{lx}" y="{ly}" font-size="12.5" text-anchor="{anchor}" fill="#1F2630" '
                 f'paint-order="stroke" stroke="#FFFFFF" stroke-width="4">{esc(label)}</text>')
    for row, col, path, line, small in BOXES.values():
        _, fill, border = ROWS[row]
        x, y = COL[col] - BW // 2, ROW[row]
        o.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" fill="{fill}" stroke="{border}" stroke-width="1.6"/>')
        o.append(f'<text x="{x + BW // 2}" y="{y + 22}" font-size="13.5" font-weight="700" text-anchor="middle" fill="#1F2630">{esc(path)}</text>')
        o.append(f'<text x="{x + BW // 2}" y="{y + 42}" font-size="12.5" text-anchor="middle" fill="#2F3742">{esc(line)}</text>')
        o.append(f'<text x="{x + BW // 2}" y="{y + 59}" font-size="11" font-style="italic" text-anchor="middle" fill="#6B7380">{esc(small)}</text>')
    for y, text, fill, dash in ((898, FOUNDATION, "#F1F0EC", ""), (968, OUTSIDE, "#FFFFFF", ' stroke-dasharray="5 4"')):
        o.append(f'<rect x="30" y="{y}" width="{W - 60}" height="50" rx="12" fill="{fill}" stroke="#8A887F" stroke-width="1.2"{dash}/>')
        o.append(f'<text x="{W // 2}" y="{y + 30}" font-size="13.5" text-anchor="middle" fill="#2F3742">{esc(text)}</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    svg_path = OUT / "code_map.svg"
    svg_path.write_text(svg())
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={W},{H}", f"--screenshot={OUT / 'code_map.png'}", f"file://{svg_path}"],
                   check=True, capture_output=True)
    (OUT / "code_map.dot").unlink(missing_ok=True)
    print("wrote", svg_path, "and", OUT / "code_map.png")
