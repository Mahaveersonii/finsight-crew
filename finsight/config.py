"""Central configuration, read once from environment / .env."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .markets import MARKETS

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Market ----------------------------------------------------------------
MARKET = os.getenv("MARKET", "US").strip().upper()
if MARKET not in MARKETS:
    raise ValueError(f"MARKET must be one of {sorted(MARKETS)}, got {MARKET!r}")
M = MARKETS[MARKET]

DATA_DIR = Path(os.getenv("FINSIGHT_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM -------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Ordered fallback chain. Each entry is a LiteLLM model string.
# Cloud models are only used when their API key is present.
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "ollama/finsight-qwen3")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("FALLBACK_MODELS", "ollama/qwen3:8b,ollama/llama3.2:latest").split(",")
    if m.strip()
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Storage ---------------------------------------------------------------
# Postgres when available (docker compose); SQLite file otherwise.
# Each market has its own database so dollar and rupee portfolios never mix.
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL and M["db_suffix"] and not DATABASE_URL.endswith(M["db_suffix"]):
    DATABASE_URL = re.sub(r"/([A-Za-z0-9_]+)$", lambda m: "/" + m.group(1) + M["db_suffix"], DATABASE_URL)
SQLITE_URL = f"sqlite:///{DATA_DIR / ('finsight' + M['db_suffix'] + '.db')}"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- SEC -------------------------------------------------------------------
# SEC asks every client to identify itself with a name + email.
# SEC rejects anonymous clients, so a blank value falls back to a generic, non-personal identity.
SEC_IDENTITY = os.getenv("SEC_IDENTITY", "").strip() or "FinSight Crew student-project@example.com"

# --- Paper trading ---------------------------------------------------------
STARTING_CASH = float(os.getenv(f"STARTING_CASH_{MARKET}", M["starting_cash"]))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))      # 1% of equity at risk per trade
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))  # max 10% of equity in one stock
MAX_SECTOR_PCT = float(os.getenv("MAX_SECTOR_PCT", "0.30"))      # max 30% of equity in one sector
MIN_POSITION_PCT = float(os.getenv("MIN_POSITION_PCT", "0.01"))  # skip trades smaller than 1% of equity
ATR_STOP_MULT = float(os.getenv("ATR_STOP_MULT", "2.0"))         # stop = entry - 2 x ATR(14)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))      # below this a BUY becomes HOLD

WATCHLIST = [
    t.strip().upper()
    for t in os.getenv(f"WATCHLIST_{MARKET}", ",".join(M["watchlist"])).split(",")
    if t.strip()
]

# Links between the two Streamlit instances (the market switch in the sidebar).
APP_URLS = {"US": os.getenv("APP_URL_US", "http://localhost:8502"), "IN": os.getenv("APP_URL_IN", "http://localhost:8503")}


def normalize_ticker(ticker: str) -> str:
    """'wipro' -> 'WIPRO.NS' in the India market; US tickers pass through."""
    t = str(ticker).strip().strip("'\"").upper().split()[0] if str(ticker).strip() else ""
    suffix = M["ticker_suffix"]
    if suffix and t and "." not in t and not t.startswith("^"):
        t += suffix
    return t
