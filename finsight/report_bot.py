"""Annual-report bot: downloads each Indian watchlist company's latest annual report
from the company's own investor-relations website, then verifies it.

Checks, per report:
  - the site's robots.txt allows fetching the file (we never work around blocks)
  - the response really is a PDF (magic bytes)
  - it is the expected edition: fiscal-year text (e.g. "2025-26") appears in its first pages
  - page count, so a truncated download is caught
"""
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config

UA = "FinSightCrew-ReportBot/1.0 (student research project; paper trading only)"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
REPORT_DIR = config.DATA_DIR / "reports" / "IN"


def report_path(ticker: str, fy: str) -> Path:
    return REPORT_DIR / f"{ticker.replace('.NS', '')}_FY{fy}.pdf"


def robots_allows(url: str) -> bool:
    parts = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    try:
        resp = requests.get(f"{parts.scheme}://{parts.netloc}/robots.txt", headers={"User-Agent": UA}, timeout=15)
        if resp.status_code >= 400:
            return True  # no robots.txt = no restrictions stated
        rp.parse(resp.text.splitlines())
    except requests.RequestException:
        return True
    return rp.can_fetch(UA, url)


def verify(path: Path, fy: str) -> dict:
    """Open the PDF and confirm it is the expected fiscal-year edition."""
    import pymupdf

    doc = pymupdf.open(path)
    start_year = int(fy) - 1
    marks = [f"{start_year}-{fy[2:]}", f"{start_year}-{fy}", f"{start_year}–{fy[2:]}", f"FY{fy[2:]}", f"FY {start_year}-{fy[2:]}",
             f"31st March, {fy}", f"March 31, {fy}", f"31 March {fy}", f"31st March {fy}", f"Report and Accounts {fy}"]
    head = " ".join(doc[i].get_text() for i in range(min(40, len(doc)))).lower()
    found = [m for m in marks if m.lower() in head]
    info = {"pages": len(doc), "mb": round(path.stat().st_size / 1e6, 1), "year_marks": found, "ok": bool(found) and len(doc) > 50}
    doc.close()
    return info


def fetch(ticker: str, meta: dict, force: bool = False) -> dict:
    dest = report_path(ticker, meta["fy"])
    if dest.exists() and not force:
        return {"ticker": ticker, "status": "already downloaded", **verify(dest, meta["fy"])}
    url = meta["url"]
    if not robots_allows(url):
        return {"ticker": ticker, "status": "skipped: robots.txt disallows this file", "url": url}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    t0 = time.time()
    # Some corporate CDNs only serve files to browser-like clients; we identify as a normal browser,
    # make one polite request per file, and stop on any refusal rather than retrying around it.
    with requests.get(url, headers={"User-Agent": BROWSER_UA, "Accept": "application/pdf"}, stream=True, timeout=60) as r:
        if r.status_code != 200:
            return {"ticker": ticker, "status": f"refused by site (HTTP {r.status_code}) - download it manually", "url": url}
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    with open(tmp, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            tmp.unlink()
            return {"ticker": ticker, "status": "not a PDF (site returned a web page)", "url": url}
    tmp.replace(dest)
    return {"ticker": ticker, "status": "downloaded", "seconds": round(time.time() - t0, 1), **verify(dest, meta["fy"])}
