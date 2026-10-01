"""Exchange hours for the active market (NSE for India, NYSE for the US).

Weekends are handled; exchange holidays are not (a holiday simply looks "open"
but prices do not move, so a queued order fills at the next real session).
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config

EXCHANGE = {"US": "NYSE", "IN": "NSE"}[config.MARKET]
TZ = ZoneInfo(config.M["timezone"])
TZ_LABEL = {"America/New_York": "New York time", "Asia/Kolkata": "IST"}[config.M["timezone"]]


def _minutes(hh_mm):
    return hh_mm[0] * 60 + hh_mm[1]


def now():
    return datetime.now(TZ)


def is_open(at=None, grace_min: int = 0) -> bool:
    """True during the regular session (optionally plus `grace_min` minutes after the close)."""
    at = (at or now()).astimezone(TZ)
    open_m, close_m = (_minutes(t) for t in config.M["market_hours"])
    m = at.hour * 60 + at.minute
    return at.weekday() < 5 and open_m <= m <= close_m + grace_min


def next_open(at=None) -> datetime:
    at = (at or now()).astimezone(TZ)
    (oh, om), _ = config.M["market_hours"]
    candidate = at.replace(hour=oh, minute=om, second=0, microsecond=0)
    if candidate <= at:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def close_today(at=None) -> datetime:
    at = (at or now()).astimezone(TZ)
    _, (ch, cm) = config.M["market_hours"]
    return at.replace(hour=ch, minute=cm, second=0, microsecond=0)


def status(at=None) -> dict:
    """Human-readable market status for the UI."""
    at = (at or now()).astimezone(TZ)
    if is_open(at):
        return {"open": True, "label": f"{EXCHANGE} open", "detail": f"closes {close_today(at):%-I:%M %p} {TZ_LABEL}"}
    nxt = next_open(at)
    day = "today" if nxt.date() == at.date() else ("tomorrow" if nxt.date() == (at + timedelta(days=1)).date()
                                                   else f"{nxt:%a %d %b}")
    return {"open": False, "label": f"{EXCHANGE} closed", "detail": f"opens {day} {nxt:%-I:%M %p} {TZ_LABEL}"}
