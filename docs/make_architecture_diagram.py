"""Generate docs/architecture.svg (used by the README) from one layout definition.

The same layout is used for the diagram in the team guide, so both always match.

    python docs/make_architecture_diagram.py
"""
from pathlib import Path

W, H = 760, 856
TITLE = "Every trade: data in, three AI agents, guardrails, then the risk officer"
SUBTITLE = "FinSight Crew - the same pipeline runs for the US and India markets"

# (anchor, x, y, w, h, style, name, lines)   style: box | accent | soft
BOXES = [
    ("yahoo", 24, 76, 220, 72, "box", "Yahoo Finance", ["live prices, company numbers", "both markets"]),
    ("sec", 270, 76, 220, 72, "box", "SEC EDGAR (US)", ["10-K annual reports", "backup company numbers"]),
    ("reportbot", 516, 76, 220, 72, "box", "Report bot (India)", ["annual-report PDFs", "checked: robots, PDF, FY26"]),
    ("library", 270, 176, 466, 56, "box", "Report library (RAG)", ["US 1,523 + India 2,187 passages in ChromaDB, cited to the page"]),
    ("agent1", 40, 280, 208, 88, "box", "1. Data Extractor", ["collects price and trend", "revenue, profit, debt", "writes the data brief"]),
    ("agent2", 276, 280, 208, 88, "box", "2. Financial Analyst", ["values the share (DCF)", "quotes the annual report", "verdict and score 0-100"]),
    ("agent3", 512, 280, 208, 88, "box", "3. Portfolio Manager", ["decides BUY, HOLD or SELL", "confidence and share count", "checks the portfolio first"]),
    ("scheduler", 40, 468, 208, 88, "box", "Schedulers (US, India)", ["prices: every 15 min", "in market hours", "crew: weekdays after close"]),
    ("risk", 276, 468, 208, 88, "accent", "Risk officer (code)", ["enforces every fund rule", "sets stop-loss and size", "score under 70 means HOLD"]),
    ("guardrails", 512, 468, 208, 88, "soft", "Guardrails (code)", ["format and tool use checked", "citations must be real", "numbers fact-checked"]),
    ("database", 276, 596, 208, 72, "box", "PostgreSQL", ["one database per market", "trades, positions, logs"]),
    ("you", 40, 712, 208, 72, "box", "You", ["type a ticker, press Run", "or ask the report"]),
    ("streamlit", 276, 712, 208, 72, "box", "Streamlit website", ["US :8502 - India :8503", "switch button in sidebar"]),
    ("grafana", 512, 712, 208, 72, "box", "Grafana dashboards", ["Market dropdown: US / India", "fund, tool calls, recoveries"]),
]
CONTAINER = ("ai-team", 24, 264, 712, 164, "The AI team (CrewAI) - Qwen3 8B on Ollama, backups qwen3:8b and llama3.2", 412)
# (path, has arrowhead)
CONNECTORS = [
    "M134 148V278", "M380 148V174", "M626 148V174", "M380 232V278",
    "M248 324H274", "M484 324H510", "M656 368V466", "M552 468V370",
    "M512 512H486", "M144 468V430", "M248 512H274", "M380 556V594",
    "M380 668V710", "M484 632H616V710", "M248 748H274",
]
# (id, x, y, text, anchor)
LABELS = [
    ("label-facts", 142, 214, "facts", "start"),
    ("label-search", 388, 256, "report search", "start"),
    ("label-decision", 648, 452, "decision card", "end"),
    ("label-retry", 544, 452, "retry if wrong", "end"),
    ("label-approved", 499, 504, "ok", "middle"),
    ("label-starts", 152, 452, "starts runs", "start"),
]
FOOT = ["Docker runs everything except Ollama, which uses the Mac's graphics chip.",
        "Each market has its own database, scheduler and Streamlit app."]

LIGHT = {"ink": "#1f2328", "quiet": "#57606a", "edge": "#8c959f", "accent": "#0969da",
         "accent_fill": "#ddf4ff", "soft": "#9a6700", "soft_fill": "#fff8c5", "tint": "#f6f8fa", "bg": "#ffffff"}


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(c=LIGHT) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="13" role="img" aria-label="{esc(TITLE)}">',
           f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
           f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
           f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c["edge"]}"/></marker></defs>',
           f'<text x="24" y="32" font-size="15" font-weight="600" fill="{c["ink"]}">{esc(TITLE)}</text>',
           f'<text x="24" y="52" font-size="11.5" fill="{c["quiet"]}">{esc(SUBTITLE)}</text>']
    _, x, y, w, h, name, ny = CONTAINER
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{c["tint"]}" stroke="{c["edge"]}" stroke-width="1.25"/>')
    out.append(f'<text x="{x + 16}" y="{ny}" font-weight="600" fill="{c["ink"]}">{esc(name)}</text>')
    for d in CONNECTORS:
        out.append(f'<path d="{d}" fill="none" stroke="{c["edge"]}" stroke-width="1.25" marker-end="url(#arrow)"/>')
    for _, x, y, w, h, style, name, lines in BOXES:
        stroke = {"box": c["edge"], "accent": c["accent"], "soft": c["soft"]}[style]
        fill = {"box": c["bg"], "accent": c["accent_fill"], "soft": c["soft_fill"]}[style]
        sw = 1.25 if style == "box" else 2
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        out.append(f'<text x="{x + 16}" y="{y + 24}" font-weight="600" fill="{c["ink"]}">{esc(name)}</text>')
        for i, ln in enumerate(lines):
            col = c["ink"] if style != "box" else c["quiet"]
            out.append(f'<text x="{x + 16}" y="{y + 40 + 16 * i}" font-size="11.5" fill="{col}">{esc(ln)}</text>')
    for _, x, y, t, anchor in LABELS:
        out.append(f'<text x="{x}" y="{y}" font-size="11.5" text-anchor="{anchor}" fill="{c["quiet"]}">{esc(t)}</text>')
    for i, t in enumerate(FOOT):
        out.append(f'<text x="24" y="{H - 40 + 16 * i}" font-size="11.5" fill="{c["quiet"]}">{esc(t)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def widget_jsx() -> str:
    """The same drawing as a doc widget module (theme tokens, one editable id per label)."""
    parts = ["export default () => { const edge = 'var(--cds-chart-axis)', accent = 'var(--cds-chart-categorical-1)', "
             "soft = 'var(--cds-chart-status-warning)', ink = 'var(--cds-text-primary)', quiet = 'var(--cds-text-secondary)'; "
             "return <svg viewBox='0 0 %d %d' role='img' aria-label='%s' fontSize='13'>" % (W, H, TITLE.replace("'", "")),
             "<defs><marker id='arch2-arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' "
             "orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill={edge}/></marker></defs>",
             "<text data-claude-text-id='title' x='24' y='32' fontSize='15' fontWeight='600' fill={ink}>%s</text>" % TITLE,
             "<text data-claude-text-id='subtitle' x='24' y='52' fontSize='11.5' fill={quiet}>%s</text>" % SUBTITLE]
    a, x, y, w, h, name, ny = CONTAINER
    parts.append(f"<g data-claude-anchor='{a}'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' "
                 f"fill='var(--cds-chart-reference-tint)' stroke={{edge}} strokeWidth='1.25'/>"
                 f"<text data-claude-text-id='team-name' x='{x + 16}' y='{ny}' fontWeight='600' fill={{ink}}>{name}</text></g>")
    parts.append("<g data-claude-anchor='connectors' fill='none' stroke={edge} strokeWidth='1.25'>"
                 + "".join(f"<path d='{d}' markerEnd='url(#arch2-arrow)'/>" for d in CONNECTORS) + "</g>")
    for anchor, x, y, w, h, style, name, lines in BOXES:
        col = {"box": "edge", "accent": "accent", "soft": "soft"}[style]
        fill = "'none'" if style == "box" else "{%s}" % col
        extra = "" if style == "box" else " fillOpacity='0.12'"
        sw = "1.25" if style == "box" else "2"
        g = [f"<g data-claude-anchor='{anchor}'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill={fill}{extra} "
             f"stroke={{{col}}} strokeWidth='{sw}'/>",
             f"<text data-claude-text-id='{anchor}-name' x='{x + 16}' y='{y + 24}' fontWeight='600' fill={{ink}}>{name}</text>"]
        for i, ln in enumerate(lines):
            tone = "quiet" if style == "box" else "ink"
            g.append(f"<text data-claude-text-id='{anchor}-l{i + 1}' x='{x + 16}' y='{y + 40 + 16 * i}' fontSize='11.5' "
                     f"fill={{{tone}}}>{ln}</text>")
        parts.append("".join(g) + "</g>")
    for tid, x, y, t, anchor in LABELS:
        parts.append(f"<text data-claude-text-id='{tid}' x='{x}' y='{y}' fontSize='11.5' textAnchor='{anchor}' fill={{quiet}}>{t}</text>")
    for i, t in enumerate(FOOT):
        parts.append(f"<text data-claude-text-id='foot-{i + 1}' x='24' y='{H - 40 + 16 * i}' fontSize='11.5' fill={{quiet}}>{t}</text>")
    parts.append("</svg>; };")
    return "".join(parts)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "architecture.svg"
    out.write_text(svg())
    print(f"wrote {out}")
