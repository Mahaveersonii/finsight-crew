"""RAG over company annual reports.

Corpus (per market, see markets.py)
  US    SEC EDGAR 10-K via edgartools -> Item 1 Business, Item 1A Risk Factors, Item 7 MD&A
  India company annual-report PDF (report_bot.py) -> narrative pages, labelled
        Business & Strategy / Risk Management / MD&A / Board's Report (pdf_reports.py)

Ingestion
  -> paragraph-aware chunks (~1,200 chars, 200-char overlap)
  -> nomic-embed-text via Ollama ("search_document:" prefix)
  -> ChromaDB (persistent, cosine distance), one collection for all tickers

Retrieval (hybrid)
  query -> nomic-embed-text ("search_query:" prefix) -> top-25 by cosine,
  filtered to the ticker -> re-ranked by 0.75 x semantic + 0.25 x keyword overlap
  -> top-k chunks returned with a citation string, e.g. [AAPL 10-K FY2025 · Risk Factors · #12]
     or, for an Indian annual report, [WIPRO.NS AR FY2026 · Risk Management · p57 #212]
"""
import hashlib
import logging
import os
import re
import time

import chromadb
import ollama

from . import config, db

log = logging.getLogger(__name__)

SECTIONS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
}
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200
PDF_CHUNK_CHARS = 1200
CONTEXT_HEADERS = os.getenv("RAG_CONTEXT_HEADERS", "1") == "1"
COLLECTION = config.M["collection"]

_client = None


def collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _ollama():
    return ollama.Client(host=config.OLLAMA_BASE_URL)


def embed(texts, kind="document"):
    prefix = "search_document: " if kind == "document" else "search_query: "
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        out.extend(_ollama().embed(model=config.EMBED_MODEL, input=batch)["embeddings"])
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _tail(s: str, n: int) -> str:
    """Last ~n chars of s, starting on a sentence (or at least word) boundary."""
    t = s[-n:]
    i = t.find(". ")
    if 0 <= i < n // 2:
        return t[i + 2:]
    j = t.find(" ")
    return t[j + 1:] if j >= 0 else t


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS):
    """Pack paragraphs into ~size-char chunks; carry the last `overlap` chars
    into the next chunk so facts that straddle a boundary stay retrievable."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph: hard split on sentence boundary
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            pieces = (cur + "\n\n" + p[:cut]).strip() if cur else p[:cut]
            chunks.append(pieces)
            cur = _tail(pieces, overlap)
            p = p[cut:].strip()
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = (_tail(cur, overlap) + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _section_text(tenk, item: str) -> str:
    for getter in (lambda: tenk[item], lambda: tenk.get_item_with_part(item) if hasattr(tenk, "get_item_with_part") else None):
        try:
            txt = getter()
            if txt and len(str(txt)) > 200:
                return str(txt)
        except Exception:  # noqa: BLE001
            continue
    return ""


def ingest_ticker(ticker: str, n_filings: int = 1, force: bool = False) -> dict:
    """Download the latest annual report for `ticker` and index it. Idempotent."""
    ticker = config.normalize_ticker(ticker)
    col = collection()
    if not force and col.get(where={"ticker": ticker}, limit=1)["ids"]:
        return {"ticker": ticker, "status": "already indexed",
                "chunks": len(col.get(where={"ticker": ticker}, include=[])["ids"])}
    if force:
        col.delete(where={"ticker": ticker})
    if config.M["corpus"] == "pdf":
        return _ingest_pdf(ticker)
    return _ingest_sec(ticker, n_filings)


def _ingest_pdf(ticker: str) -> dict:
    from .pdf_reports import extract_sections
    from .report_bot import fetch, report_path

    meta = config.M.get("reports", {}).get(ticker)
    if not meta:
        raise ValueError(f"No annual report configured for {ticker}; add it to markets.py")
    path = report_path(ticker, meta["fy"])
    if not path.exists():
        res = fetch(ticker, meta)
        if not path.exists():
            raise RuntimeError(f"Annual report for {ticker} unavailable: {res.get('status')}")
    t0 = time.time()
    docs, metas, ids = [], [], []
    size = int(os.getenv("RAG_PDF_CHUNK", PDF_CHUNK_CHARS))
    for section, page, text in extract_sections(path):
        for piece in chunk_text(text, size=size, overlap=size // 6):
            n = len(docs)
            docs.append(piece)
            ids.append(hashlib.md5(f"{ticker}|{meta['fy']}|{page}|{n}".encode()).hexdigest())
            metas.append({"ticker": ticker, "form": "AR", "fiscal_year": meta["fy"], "section": section,
                          "page": page, "chunk": n, "filing_date": "", "url": meta["url"], "title": meta["title"]})
    col = collection()
    for i in range(0, len(docs), 128):
        batch_meta = metas[i:i + 128]
        # Contextual header: the embedding sees "who / which report / which section" as well as the passage,
        # so a short PDF fragment still carries its context. The stored text stays the passage itself.
        to_embed = [f"{meta['title']} · {m['section']} · page {m['page']}\n{d}" if CONTEXT_HEADERS else d
                    for m, d in zip(batch_meta, docs[i:i + 128])]
        col.upsert(ids=ids[i:i + 128], documents=docs[i:i + 128], metadatas=batch_meta,
                   embeddings=embed(to_embed, "document"))
    db.log_event("rag_ingest", ticker, f"{len(docs)} chunks", duration_ms=(time.time() - t0) * 1000)
    return {"ticker": ticker, "status": "indexed", "chunks": len(docs), "seconds": round(time.time() - t0, 1)}


def citation(meta: dict) -> str:
    if meta.get("form") == "AR":
        return f"[{meta['ticker']} AR FY{meta['fiscal_year']} · {meta['section']} · p{meta['page']} #{meta['chunk']}]"
    return f"[{meta['ticker']} 10-K FY{meta['fiscal_year']} · {meta['section']} · #{meta['chunk']}]"


def _ingest_sec(ticker: str, n_filings: int = 1) -> dict:
    from edgar import Company, set_identity

    col = collection()
    set_identity(config.SEC_IDENTITY)
    t0 = time.time()
    filings = Company(ticker).get_filings(form="10-K").head(n_filings)
    total = 0
    for filing in filings:
        tenk = filing.obj()
        fy = str(getattr(filing, "period_of_report", "") or filing.filing_date)[:4]
        url = getattr(filing, "filing_url", None) or getattr(filing, "url", "")
        for item, label in SECTIONS.items():
            body = _section_text(tenk, item)
            if not body:
                log.warning("%s %s: section %s empty", ticker, fy, item)
                continue
            chunks = chunk_text(body)
            ids = [hashlib.md5(f"{ticker}|{filing.accession_no}|{item}|{i}".encode()).hexdigest() for i in range(len(chunks))]
            metas = [{"ticker": ticker, "form": "10-K", "fiscal_year": fy, "section": label, "item": item,
                      "chunk": i, "filing_date": str(filing.filing_date), "url": url} for i in range(len(chunks))]
            col.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embed(chunks, "document"))
            total += len(chunks)
    db.log_event("rag_ingest", ticker, f"{total} chunks", duration_ms=(time.time() - t0) * 1000)
    return {"ticker": ticker, "status": "indexed", "chunks": total, "seconds": round(time.time() - t0, 1)}


def indexed_tickers() -> dict:
    metas = collection().get(include=["metadatas"])["metadatas"]
    counts = {}
    for m in metas:
        counts[m["ticker"]] = counts.get(m["ticker"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z][a-z\-]{2,}")
_STOP = set("the and for with that this from are was were has have its their our any not but into over under than which what when how does about company".split())


def _keywords(s):
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


def search(ticker: str, query: str, k: int = 4, section: str = None, run_id=None, alpha: float = None,
           log: bool = True) -> list:
    t0 = time.time()
    ticker = config.normalize_ticker(ticker)
    alpha = config.M["rag_alpha"] if alpha is None else alpha
    where = {"ticker": ticker}
    if section:
        where = {"$and": [where, {"section": section}]}
    res = collection().query(query_embeddings=embed([query], "query"), n_results=25, where=where,
                             include=["documents", "metadatas", "distances"])
    if not res["ids"] or not res["ids"][0]:
        return []
    q_terms = _keywords(query)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        semantic = 1 - dist
        overlap = len(q_terms & _keywords(doc)) / max(len(q_terms), 1)
        hits.append({
            "score": round(alpha * semantic + (1 - alpha) * overlap, 3),
            "semantic": round(semantic, 3),
            "citation": citation(meta),
            "page": meta.get("page"),
            "section": meta["section"],
            "fiscal_year": meta["fiscal_year"],
            "text": doc,
            "url": meta.get("url", ""),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    if log:
        db.log_event("rag_query", ticker.upper(), query, run_id, (time.time() - t0) * 1000)
    return hits[:k]


# ---------------------------------------------------------------------------
# Grounded Q&A ("chat with the 10-K")
# ---------------------------------------------------------------------------

def answer(ticker: str, question: str, k: int = 4) -> dict:
    """Retrieve, then answer strictly from the retrieved passages with citations."""
    hits = search(ticker, question, k=k)
    if not hits:
        return {"answer": f"No indexed filings for {ticker}. Index it first.", "sources": []}
    context = "\n\n".join(f"{h['citation']}\n{h['text']}" for h in hits)
    prompt = (
        f"Answer the question about {ticker} using ONLY the {config.M['doc_name']} passages below. "
        "After each claim put the citation tag it came from, exactly as written above the passage. "
        "If the passages do not contain the answer, say so plainly.\n\n"
        f"PASSAGES:\n{context}\n\nQUESTION: {question}\nANSWER (max 150 words):"
    )
    from .llm import make_llm, model_chain  # local import: llm imports crewai, which is slow to load

    t0 = time.time()
    text, last_err = None, None
    for model in model_chain():  # same fallback chain as the agents (Groq, then any other configured model)
        try:
            text = str(make_llm(model).call(prompt)).strip()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            db.log_event("llm_fallback", model, f"rag_answer: {type(exc).__name__}: {exc}")
    if text is None:
        raise RuntimeError(f"No AI model could answer: {last_err}")
    db.log_event("rag_answer", ticker.upper(), question, duration_ms=(time.time() - t0) * 1000)
    return {"answer": text, "sources": hits}
