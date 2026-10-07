"""Generate docs/architecture.svg (used by the README) from one layout definition.

The same layout is used for the diagram in the team guide, so both always match.

    python docs/make_architecture_diagram.py
"""
from pathlib import Path

W, H = 900, 860
TITLE = "Every decision: data in, three AI agents on Groq, code checks the facts, then the risk officer"
SUBTITLE = "FinSight Crew v2 - the same pipeline runs for the US and India markets - NEW marks what changed in v2"

# (anchor, x, y, w, h, style, name, lines)   style: box | accent | soft
BOXES = [
    ("yahoo", 24, 76, 200, 72, "box", "Yahoo Finance", ["prices refreshed every minute", "company numbers, both markets"]),
    ("sec", 244, 76, 200, 72, "box", "SEC EDGAR (US)", ["10-K annual reports", "backup company numbers"]),
    ("reportbot", 464, 76, 200, 72, "box", "Report bot (India)", ["annual-report PDFs", "checked: robots, PDF, FY26"]),
    ("library", 244, 176, 420, 56, "box", "Report library (RAG)", ["10-K and annual-report passages in ChromaDB, cited to the page"]),
    ("ollama", 700, 176, 176, 56, "box", "Ollama (Mac)", ["turns text into vectors"]),
    ("groq", 700, 280, 176, 88, "accent", "Groq cloud AI", ["gpt-oss-120b (main)", "backups: 20b, Qwen 3.8", "200K tokens a day each"]),
    ("agent1", 40, 280, 190, 88, "box", "1. Data Extractor", ["price, trend and", "company numbers", "writes the data brief"]),
    ("agent2", 258, 280, 190, 88, "box", "2. Financial Analyst", ["values the share (DCF)", "quotes the annual report", "verdict and score 0-100"]),
    ("agent3", 476, 280, 190, 88, "box", "3. Portfolio Manager", ["gets an exact fact sheet", "decides BUY, HOLD or SELL", "hands in a JSON decision"]),
    ("scheduler", 40, 468, 190, 88, "box", "Schedulers (US, India)", ["every 15 min: prices, stops", "after the close: agents", "run on the Mac"]),
    ("risk", 258, 468, 190, 88, "accent", "Risk officer (code)", ["score under 70 means HOLD", "sets stop-loss and size", "10% stock, 30% sector caps"]),
    ("guardrails", 476, 468, 190, 88, "soft", "Guardrails (code)", ["format and tool use checked", "citations matched or fixed", "numbers fact-checked"]),
    ("broker", 258, 596, 190, 72, "box", "Paper broker (code)", ["fills while market is open", "queues orders when closed"]),
    ("database", 476, 596, 190, 72, "box", "PostgreSQL (Docker)", ["one database per market", "decisions, trades, audit log"]),
    ("where", 700, 596, 176, 188, "soft", "Where it runs", ["Mac: both apps, the", "schedulers and Ollama", "Docker: PostgreSQL and", "Grafana", "Groq cloud: AI models", "Internet: Yahoo, SEC,", "company websites", "All on free plans"]),
    ("you", 40, 712, 190, 72, "box", "You", ["pick a stock, press Run", "or ask the report"]),
    ("streamlit", 258, 712, 190, 72, "box", "Streamlit app", ["trading desk, live prices", "US :8502 - India :8503"]),
    ("grafana", 476, 712, 190, 72, "box", "Grafana (Docker)", ["Market dropdown: US / India", "fund and agent health"]),
]
NEW = {"ollama", "groq", "agent3", "guardrails", "streamlit"}
CONTAINER = ("ai-team", 24, 264, 656, 164, "The AI team (CrewAI) - runs on Groq and switches model if one fails", 412)
# (path, has arrowhead)
CONNECTORS = [
    "M124 148V278", "M344 148V174", "M564 148V174", "M700 204H666", "M353 232V278",
    "M230 324H256", "M448 324H474", "M700 324H682", "M620 368V466", "M520 468V370",
    "M476 512H450", "M135 468V430", "M230 512H256", "M353 556V594", "M448 632H474",
    "M620 668V710", "M520 668V690H353V710", "M230 748H256",
]
# (id, x, y, text, anchor)
LABELS = [
    ("label-facts", 132, 214, "facts", "start"),
    ("label-search", 361, 256, "report search", "start"),
    ("label-decision", 612, 452, "decision", "end"),
    ("label-retry", 512, 452, "retry if wrong", "end"),
    ("label-approved", 463, 504, "ok", "middle"),
    ("label-starts", 143, 452, "starts runs", "start"),
]
FOOT = ["v2: the AI moved from the Mac to Groq's cloud, code hands the agents exact facts and repairs citations,",
        "and the screens became a live trading desk. Each market has its own database, scheduler and app."]

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
    for anchor, x, y, w, h, *_ in BOXES:
        if anchor in NEW:
            out.append(f'<rect x="{x + w - 46}" y="{y + 10}" width="36" height="16" rx="4" fill="{c["accent"]}"/>'
                       f'<text x="{x + w - 28}" y="{y + 22}" font-size="10" font-weight="600" text-anchor="middle" '
                       f'fill="{c["bg"]}">NEW</text>')
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
        if anchor in NEW:
            g.append(f"<rect x='{x + w - 46}' y='{y + 10}' width='36' height='16' rx='4' fill={{accent}} fillOpacity='0.15' stroke={{accent}}/>"
                     f"<text data-claude-text-id='{anchor}-new' x='{x + w - 28}' y='{y + 22}' fontSize='10' fontWeight='600' "
                     f"textAnchor='middle' fill={{accent}}>NEW</text>")
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
