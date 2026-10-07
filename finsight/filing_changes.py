"""Filing Change Analyst: what changed in a company's 10-K since last year.

Research basis: Cohen, Malloy & Nguyen (2020), "Lazy Prices", Journal of Finance 75(3).
Firms that change the language of their annual report - especially Risk Factors - later
underperform, and investors are slow to notice because nobody re-reads 100 pages a year.

Division of labour (same principle as the rest of v2: the AI judges, code supplies facts):
  code   -> pull the same sections from the last two 10-Ks, match every paragraph to its
            closest one last year (meaning first, then word-by-word), measure how much changed
  agent  -> read only the new, removed and edited paragraphs and explain what they mean,
            citing the paragraph tags it was given (checked by a guardrail)
Results are cached per pair of filings: a 10-K changes once a year.
"""
import difflib
import json
import logging
import math
import re
from collections import Counter

import numpy as np

from . import config, db, rag

log = logging.getLogger(__name__)

def _loose(word: str) -> str:
    """Match a word even when the HTML split it across lines ("RIS\\nK FACTORS" in one Microsoft 10-K)."""
    return r"\s*".join(map(re.escape, word))


# section label -> (heading where it starts, heading where the next section starts)
SECTIONS = {
    "Risk Factors": (r"^\W*item\s*1a\W+" + _loose("risk") + r"\s*" + _loose("factors"), r"^\W*item\s*(1b|1c|2)\W"),
    "MD&A": (r"^\W*item\s*7\W+" + _loose("management"), r"^\W*item\s*7a\W"),
}
ITEM_OF = {"Risk Factors": "Item 1A", "MD&A": "Item 7"}
MIN_PARA = 80          # shorter lines are headings, page numbers or table cells
SAME, EDITED = 0.97, 0.60   # word-level similarity: >= SAME unchanged, >= EDITED edited, else new
CANDIDATES = 3         # compare words against the 3 closest paragraphs by meaning
_WORD = re.compile(r"[a-z][a-z'\-]+")


# ---------------------------------------------------------------------------
# Getting the text
# ---------------------------------------------------------------------------
def _plain(raw: str) -> str:
    if raw.lstrip()[:1] == "<":  # some filings come back as inline-XBRL HTML, not text
        from bs4 import BeautifulSoup
        raw = BeautifulSoup(raw, "lxml").get_text("\n")
    raw = raw.replace("\xa0", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    return re.sub(r"\n\s*\n+", "\n\n", raw)


def cut_section(text: str, start_pat: str, end_pat: str) -> str:
    """Longest span from a section heading to the next one, skipping table-of-contents entries
    (a TOC entry is recognisable because another start heading appears before it ends)."""
    starts = [m.start() for m in re.finditer(start_pat, text, re.I | re.M)]
    end_re = re.compile(end_pat, re.I | re.M)
    best = ""
    for s in starts:
        m = end_re.search(text, s + 200)
        if not m or any(s < s2 < m.start() for s2 in starts):
            continue
        if re.search(r"^\W*item\s*(1b|7a|8)\W", text[s + 20:s + 600], re.I | re.M):
            continue  # a contents page listing the next items straight after this heading
        if len(best) < m.start() - s < 400_000:
            best = text[s:m.start()]
    return best


def _sections(filing) -> dict:
    """Cut both years with the same heading rules, so the comparison is like for like. (The SEC
    library's own section finder sometimes returns the table of contents and Item 1 as well.)"""
    text, out, tenk = _plain(filing.text() or ""), {}, None
    for label, (start, end) in SECTIONS.items():
        body = cut_section(text, start, end)
        if not body:
            try:
                tenk = tenk or filing.obj()
                body = rag._section_text(tenk, ITEM_OF[label])
            except Exception:  # noqa: BLE001
                body = ""
        out[label] = body
    return out


def paragraphs(text: str) -> list:
    """Re-join lines that were wrapped mid-sentence (common in SEC HTML) into whole paragraphs."""
    paras, buf = [], ""
    for line in (ln.strip() for ln in text.split("\n")):
        if not line or re.fullmatch(r"(page\s*)?\d{1,3}|part\s+[iv]+", line, re.I):
            continue
        if buf and (not re.search(r"[.!?:;)\"\u201d]$", buf) or line[:1].islower()):
            buf += " " + line
        else:
            if buf:
                paras.append(buf)
            buf = line
    if buf:
        paras.append(buf)
    return [p for p in paras if len(p) >= MIN_PARA]


# ---------------------------------------------------------------------------
# Measuring the change
# ---------------------------------------------------------------------------
def cosine_similarity(a: str, b: str) -> float:
    """Word-count cosine similarity of two documents, the measure used in Lazy Prices (1 = identical)."""
    ca, cb = Counter(_WORD.findall(a.lower())), Counter(_WORD.findall(b.lower()))
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    return round(dot / (na * nb), 4) if na and nb else 0.0


def word_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def _embed(texts):
    m = np.array(rag.embed(texts, "document"), dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def compare(new: list, old: list, embed=_embed) -> dict:
    """Classify each new paragraph as unchanged / edited / new and each old one as kept / removed."""
    if not new or not old:
        return {"unchanged": 0, "edited": [], "added": [], "removed": []}
    sims = embed(new) @ embed(old).T

    def best_match(i_row, row_texts, col_texts, matrix):
        cand = np.argsort(matrix[i_row])[::-1][:CANDIDATES]
        scores = [(word_ratio(row_texts[i_row], col_texts[j]), int(j)) for j in cand]
        return max(scores)

    unchanged, edited, added = 0, [], []
    for i, p in enumerate(new):
        ratio, j = best_match(i, new, old, sims)
        if ratio >= SAME:
            unchanged += 1
        elif ratio >= EDITED:
            edited.append({"n": i + 1, "old_n": j + 1, "text": p, "old_text": old[j], "similarity": round(ratio, 3)})
        else:
            added.append({"n": i + 1, "text": p, "novelty": round(1 - ratio, 3)})
    removed = []
    for j, p in enumerate(old):
        ratio, _ = best_match(j, old, new, sims.T)
        if ratio < EDITED:
            removed.append({"n": j + 1, "text": p})
    added.sort(key=lambda d: -d["novelty"])
    return {"unchanged": unchanged, "edited": edited, "added": added, "removed": removed}


def word_diff(old: str, new: str) -> list:
    """(op, text) runs for showing an edited paragraph: op is 'same', 'add' or 'del'."""
    a, b = old.split(), new.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(("same", " ".join(a[i1:i2])))
        if op in ("delete", "replace"):
            out.append(("del", " ".join(a[i1:i2])))
        if op in ("insert", "replace"):
            out.append(("add", " ".join(b[j1:j2])))
    return out


def tag(ticker, fy, section, kind, n) -> str:
    return f"[{ticker} 10-K FY{fy} · {section} · {kind} ¶{n}]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _cache(ticker, acc_new, acc_old):
    return config.CACHE_DIR / f"changes_{ticker}_{acc_new}_{acc_old}.json".replace("/", "-")


def cached(ticker: str):
    """Latest saved comparison for a ticker, without contacting EDGAR (None if never run)."""
    files = sorted(config.CACHE_DIR.glob(f"changes_{config.normalize_ticker(ticker)}_*.json"),
                   key=lambda f: f.stat().st_mtime)
    return json.loads(files[-1].read_text()) if files else None


def analyse(ticker: str, explain_with_ai: bool = True, refresh: bool = False) -> dict:
    """Compare the latest two 10-Ks of a US company. Cached per pair of filings."""
    if config.MARKET != "US":
        raise ValueError("Filing comparison needs last year's report as well; it is available for US 10-Ks so far.")
    from edgar import Company, set_identity

    ticker = config.normalize_ticker(ticker)
    set_identity(config.SEC_IDENTITY)
    filings = list(Company(ticker).get_filings(form="10-K").head(2))
    if len(filings) < 2:
        raise ValueError(f"{ticker} has fewer than two 10-K filings on EDGAR")
    new_f, old_f = filings
    path = _cache(ticker, new_f.accession_no, old_f.accession_no)
    if path.exists() and not refresh:
        res = json.loads(path.read_text())
        if res.get("summary") or not explain_with_ai:
            return res
    else:
        fy = lambda f: str(getattr(f, "period_of_report", "") or f.filing_date)[:4]  # noqa: E731
        res = {"ticker": ticker, "fy_new": fy(new_f), "fy_old": fy(old_f),
               "filed_new": str(new_f.filing_date), "filed_old": str(old_f.filing_date),
               "url_new": getattr(new_f, "filing_url", ""), "url_old": getattr(old_f, "filing_url", ""),
               "sections": {}, "summary": None}
        s_new, s_old = _sections(new_f), _sections(old_f)
        for label in SECTIONS:
            pn, po = paragraphs(s_new[label]), paragraphs(s_old[label])
            if min(len(s_new[label].split()), len(s_old[label].split())) < 300 or not pn or not po:
                # e.g. banks and oil majors put MD&A in a separate exhibit; the 10-K only points to it
                res["skipped"] = {**res.get("skipped", {}), label: "not in the 10-K body (incorporated by reference)"}
                continue
            diff = compare(pn, po)
            res["sections"][label] = {
                "similarity": cosine_similarity(s_new[label], s_old[label]),
                "words_new": len(s_new[label].split()), "words_old": len(s_old[label].split()),
                "paragraphs_new": len(pn), "paragraphs_old": len(po), **diff,
            }
        db.log_event("filing_compare", ticker, json.dumps(
            {k: {"similarity": v["similarity"], "added": len(v["added"]), "removed": len(v["removed"]),
                 "edited": len(v["edited"])} for k, v in res["sections"].items()}))
    if explain_with_ai and not res.get("summary"):
        res["summary"] = explain(res)
    path.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def _evidence(res, per_kind=(8, 5, 5)) -> tuple:
    """The paragraphs the agent may read, each with the exact tag it must cite."""
    lines, tags = [], set()
    t, fn, fo = res["ticker"], res["fy_new"], res["fy_old"]
    for label, sec in res["sections"].items():
        lines.append(f"## {label}: word similarity to last year {sec['similarity']:.3f} "
                     f"({len(sec['added'])} new, {len(sec['edited'])} edited, {len(sec['removed'])} removed paragraphs)")
        for d in sec["added"][:per_kind[0]]:
            tg = tag(t, fn, label, "new", d["n"]); tags.add(tg)
            lines.append(f"{tg} NEW: {d['text'][:600]}")
        for d in sec["removed"][:per_kind[1]]:
            tg = tag(t, fo, label, "removed", d["n"]); tags.add(tg)
            lines.append(f"{tg} REMOVED: {d['text'][:400]}")
        for d in sorted(sec["edited"], key=lambda d: d["similarity"])[:per_kind[2]]:
            tg = tag(t, fn, label, "edited", d["n"]); tags.add(tg)
            lines.append(f"{tg} EDITED. Last year: {d['old_text'][:300]} | This year: {d['text'][:300]}")
    return "\n".join(lines), tags


def _check(raw: str, tags: set):
    from .crew import extract_json, norm_tag
    try:
        out = extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        return False, f"Return one JSON object only ({exc})"
    for key in ("headline", "concern", "new_risks", "removed_or_softened", "tone"):
        if key not in out:
            return False, f"JSON is missing '{key}'"
    if str(out["concern"]).title() not in {"Low", "Medium", "High"}:
        return False, "concern must be Low, Medium or High"
    out["concern"] = str(out["concern"]).title()
    real = {norm_tag(x): x for x in tags}
    for key in ("new_risks", "removed_or_softened"):
        items = out[key] if isinstance(out[key], list) else []
        kept = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cites = [real[norm_tag(c)] for c in (it.get("citations") or []) if norm_tag(c) in real]
            if cites:  # an item without a real paragraph behind it is dropped, never shown
                kept.append({**it, "citations": cites})
        out[key] = kept
    return True, out


def explain(res: dict) -> dict:
    """One CrewAI agent reads the changed paragraphs and explains them; a guardrail keeps it honest."""
    from crewai import Agent, Crew, Process, Task

    from .llm import make_llm, model_chain

    evidence, tags = _evidence(res)
    if not tags:
        return {"headline": "No material wording changes between the two reports.", "concern": "Low",
                "new_risks": [], "removed_or_softened": [], "tone": "unchanged", "model": None}

    def guard(output):
        ok, value = _check(output.raw, tags)
        if not ok:
            db.log_event("guardrail_retry", "filing_change_analyst", value)
            return False, value
        return True, json.dumps(value)

    last_err = None
    for model in model_chain():
        try:
            agent = Agent(
                role="Filing Change Analyst",
                goal=f"Explain what {res['ticker']} changed in its annual report between FY{res['fy_old']} and FY{res['fy_new']}.",
                backstory=("A forensic accounting analyst. Research (Lazy Prices, Journal of Finance 2020) shows that "
                           "changes in 10-K wording, especially new or rewritten risk factors, precede weaker results. "
                           "You only describe what the paragraphs say and always cite the tag of each paragraph."),
                llm=make_llm(model), allow_delegation=False, max_iter=3, verbose=False)
            task = Task(
                description=(
                    f"Company: {res['ticker']}. Below are ONLY the paragraphs that changed between the FY{res['fy_old']} "
                    f"and FY{res['fy_new']} 10-K, each starting with its tag.\n\n{evidence}\n\n"
                    "Return ONE JSON object with keys: headline (one sentence), concern (Low, Medium or High), tone (one sentence on whether management sounds more or less cautious), "
                    "new_risks (list of {risk, why_it_matters, citations: [tags]}), removed_or_softened (list of "
                    "{item, citations: [tags]}). Use only tags shown above, copied exactly. Max 5 items per list. "
                    "Ignore pure rewording.\n"
                    "Calibrate concern against a normal year, because every company refreshes its 10-K annually: "
                    "Low = mostly updated wording, dates and figures; Medium = one or two genuinely new risk themes; "
                    "High = several new material risks (e.g. a break-up, major litigation, a new regulator, losing a key "
                    "market) or protective language removed. Most companies in most years are Low or Medium."),
                expected_output="One JSON object, no other text.",
                agent=agent, guardrail=guard, guardrail_max_retries=2)
            out = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
            summary = json.loads(out.tasks_output[0].raw) if out.tasks_output[0].raw.strip().startswith("{") \
                else _check(out.tasks_output[0].raw, tags)[1]
            summary["model"] = model
            db.log_event("filing_explained", res["ticker"], summary.get("headline", "")[:300])
            return summary
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            db.log_event("llm_fallback", model, f"filing_change_analyst: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"No AI model could explain the changes: {last_err}")


def brief(ticker: str) -> dict:
    """Compact view for the other agents (used by the get_filing_changes tool)."""
    res = analyse(ticker)
    s = res.get("summary") or {}
    return {
        "compared": f"FY{res['fy_new']} vs FY{res['fy_old']} 10-K",
        "similarity_to_last_year": {k: v["similarity"] for k, v in res["sections"].items()},
        "paragraphs": {k: {"new": len(v["added"]), "edited": len(v["edited"]), "removed": len(v["removed"])}
                       for k, v in res["sections"].items()},
        "headline": s.get("headline"), "concern": s.get("concern"), "tone": s.get("tone"),
        "new_risks": [{"risk": r.get("risk"), "citations": r.get("citations")} for r in s.get("new_risks", [])][:4],
        "note": "Lower similarity means more rewriting; in research, big Risk Factors changes precede weaker returns.",
    }
