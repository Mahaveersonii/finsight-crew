"""Section-aware text extraction from Indian company annual reports (PDF).

Indian annual reports have no standard machine-readable structure (no XBRL text
like a US 10-K, bookmarks are often missing or meaningless), so sections are
found from page content:

  1. Keep the narrative part: pages before the audited financial statements.
  2. Drop boilerplate: corporate governance, BRSR (sustainability reporting),
     AGM notice, statutory forms and annexures.
  3. Label each remaining page: MD&A, Risk Management, Board's Report, or
     Business & Strategy (letters, strategy, business segments).

Each page keeps its printed-PDF page number so citations can point to it.
"""
import re

import pymupdf

_FIN_START = re.compile(r"independent auditor'?s'? report|balance sheet as at|standalone financial statements|"
                        r"consolidated financial statements", re.I)
_CONTENTS = re.compile(r"\bcontents\b|inside this report|what'?s inside", re.I)
_MDNA_WORDS = ("outlook", "industry", "demand", "revenue", "margin", "market share", "economy", "economic",
               "opportunit", "threat", "growth", "segment", "ebitda", "profit")
_BOILER_WORDS = {
    "brsr": ("essential indicators", "leadership indicators", "principle 1", "principle 6", "brsr"),
    "governance": ("corporate governance", "board meetings", "committee meeting", "attendance", "listing regulations"),
    "notice": ("notice is hereby", "annual general meeting", "e-voting", "remote e-voting", "proxy"),
    "forms": ("form no. aoc", "form mr-3", "secretarial", "form mgt", "din:", "lodr", "regulation 46"),
}


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"-\n(?=[a-z])", "", text)            # re-join hyphenated words
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[\d\s|/\-–]{1,8}", ln)]
    out, para = [], []
    for ln in lines:
        para.append(ln)
        if re.search(r"[.!?:]$", ln) or len(ln) < 40:
            out.append(" ".join(para))
            para = []
    if para:
        out.append(" ".join(para))
    return "\n\n".join(p for p in out if len(p) > 1)


_DIN = re.compile(r"\bDIN\b\W{0,3}\d{6,8}")   # Director Identification Number = a director biography page


def _is_boilerplate(low: str, text: str = "") -> bool:
    if _DIN.search(text):
        return True
    return any(sum(low.count(w) for w in words) >= 3 for words in _BOILER_WORDS.values())


def classify(text: str) -> str:
    """Label a narrative page by what it talks about."""
    low = text.lower()
    risk = low.count("risk") + 2 * low.count("mitigat")
    mdna = sum(low.count(w) for w in _MDNA_WORDS)
    if risk >= 6:
        return "Risk Management"
    if mdna >= 10 or "management discussion" in low[:600]:
        return "MD&A"
    return "Business & Strategy"


def extract_sections(path) -> list:
    """Return [(section_label, page_number (1-based), cleaned_text), ...] for narrative pages."""
    doc = pymupdf.open(path)
    n = len(doc)
    # The narrative part ends where the audited financials start (search after the first ~15% so a
    # contents page that *mentions* the financial statements does not end it early).
    fin_start = next((i for i in range(max(5, n // 7), n) if _FIN_START.search(doc[i].get_text()[:400])), n)
    pages = []
    for i in range(fin_start):
        text = doc[i].get_text()
        low = text.lower()
        if len(text.strip()) < 400 or _CONTENTS.search(text[:300]) or _is_boilerplate(low, text):
            continue
        pages.append((classify(text), i + 1, _clean(text)))
    doc.close()
    return pages
