"""Paper-trading broker + hard risk rules.

The agents *propose*; this module *disposes*. Every rule here is deterministic
so a hallucinating model can never breach the risk limits:
  - position size  = (equity x RISK_PER_TRADE) / (entry - stop)   ("1% risk rule")
  - capped at MAX_POSITION_PCT of equity and by available cash
  - sector exposure capped at MAX_SECTOR_PCT
  - BUY below MIN_CONFIDENCE is downgraded to HOLD
  - long-only: SELL closes an existing position, never opens a short
"""
import logging
import math

from sqlalchemy import func

from . import config, db
from .db import delete, insert, positions, select, snapshots, trades, update

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

def cash() -> float:
    with db.engine().connect() as c:
        bought = c.execute(select(func.coalesce(func.sum(trades.c.shares * trades.c.price), 0)).where(trades.c.side == "BUY")).scalar()
        sold = c.execute(select(func.coalesce(func.sum(trades.c.shares * trades.c.price), 0)).where(trades.c.side == "SELL")).scalar()
    return config.STARTING_CASH - float(bought) + float(sold)


def open_positions() -> list:
    return db.fetch_all(select(positions))


def equity() -> float:
    return cash() + sum(p["shares"] * (p["last_price"] or p["avg_price"]) for p in open_positions())


def portfolio_summary() -> dict:
    pos = open_positions()
    eq = equity()
    by_sector = {}
    for p in pos:
        v = p["shares"] * (p["last_price"] or p["avg_price"])
        by_sector[p["sector"] or "Unknown"] = by_sector.get(p["sector"] or "Unknown", 0) + v
    return {
        "cash": round(cash(), 2),
        "equity": round(eq, 2),
        "positions": [
            {"ticker": p["ticker"], "shares": p["shares"], "avg_price": round(p["avg_price"], 2),
             "stop_loss": p["stop_loss"], "weight_pct": round(p["shares"] * (p["last_price"] or p["avg_price"]) / eq * 100, 1)}
            for p in pos
        ],
        "sector_weights_pct": {k: round(v / eq * 100, 1) for k, v in by_sector.items()},
        "limits": {"risk_per_trade_pct": config.RISK_PER_TRADE * 100, "max_position_pct": config.MAX_POSITION_PCT * 100,
                   "max_sector_pct": config.MAX_SECTOR_PCT * 100, "min_confidence": config.MIN_CONFIDENCE},
    }


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def plan_trade(ticker: str, price: float, atr: float, sector: str) -> dict:
    """Compute stop-loss, take-profit and share count for a prospective BUY."""
    eq = equity()
    stop = round(price - config.ATR_STOP_MULT * atr, 2)
    risk_per_share = max(price - stop, price * 0.01)
    shares_by_risk = math.floor(eq * config.RISK_PER_TRADE / risk_per_share)
    shares_by_cap = math.floor(eq * config.MAX_POSITION_PCT / price)
    shares_by_cash = math.floor(cash() / price)

    held = next((p for p in open_positions() if p["ticker"] == ticker), None)
    held_value = held["shares"] * price if held else 0
    shares_by_cap = max(0, math.floor((eq * config.MAX_POSITION_PCT - held_value) / price))

    sector_value = sum(p["shares"] * (p["last_price"] or p["avg_price"]) for p in open_positions() if p["sector"] == sector)
    shares_by_sector = max(0, math.floor((eq * config.MAX_SECTOR_PCT - sector_value) / price))

    shares = max(0, min(shares_by_risk, shares_by_cap, shares_by_cash, shares_by_sector))
    binding = min(
        [("1% risk rule", shares_by_risk), ("max position size", shares_by_cap),
         ("available cash", shares_by_cash), ("sector cap", shares_by_sector)],
        key=lambda x: x[1],
    )[0]
    if 0 < shares * price < eq * config.MIN_POSITION_PCT:
        binding = f"{binding} (leaves < {config.MIN_POSITION_PCT:.0%} minimum position)"
        shares = 0
    return {
        "ticker": ticker, "entry_price": round(price, 2), "stop_loss": stop,
        "take_profit": round(price + 2 * (price - stop), 2),  # 2R target
        "shares": shares, "position_value": round(shares * price, 2),
        "position_pct_of_equity": round(shares * price / eq * 100, 2) if eq else 0,
        "capital_at_risk": round(shares * (price - stop), 2),
        "binding_constraint": binding, "equity": round(eq, 2), "already_held_shares": held["shares"] if held else 0,
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _fill(ticker, side, shares, price, reason, signal_id=None, realized=None):
    with db.engine().begin() as c:
        c.execute(insert(trades).values(ts=db.now(), ticker=ticker, side=side, shares=shares, price=price,
                                        reason=reason, signal_id=signal_id, realized_pnl=realized))


def execute(signal: dict, sector: str, atr: float, run_id=None, signal_id=None, queue_if_closed: bool = False) -> dict:
    """Apply risk rules to an agent signal and paper-execute it.

    With `queue_if_closed`, a BUY/SELL decided while the exchange is closed is not filled at the stale
    closing price: it is queued and filled at the first price check after the next open (fill_pending)."""
    from . import market_clock

    ticker, action, price = signal["ticker"], signal["action"], float(signal["price"])
    held = next((p for p in open_positions() if p["ticker"] == ticker), None)

    if action == "BUY" and signal.get("confidence", 0) < config.MIN_CONFIDENCE:
        db.log_event("risk_veto", "low_confidence", f"{ticker} BUY conf={signal.get('confidence')}", run_id)
        return {"status": "vetoed", "reason": f"confidence {signal.get('confidence')} < {config.MIN_CONFIDENCE}"}

    if queue_if_closed and action in ("BUY", "SELL") and not (action == "SELL" and not held) \
            and not market_clock.is_open():
        st = market_clock.status()
        reason = f"{st['label']}: order queued, it will execute at the next open ({st['detail']})"
        preview = plan_trade(ticker, price, atr, sector) if action == "BUY" else None
        db.log_event("order_queued", ticker, f"{action} {ticker} queued while {st['label']}; executes when it {st['detail']}", run_id)
        return {"status": "pending", "reason": reason, "plan": preview}

    if action == "BUY":
        plan = plan_trade(ticker, price, atr, sector)
        if plan["shares"] <= 0:
            db.log_event("risk_veto", plan["binding_constraint"], f"{ticker} BUY blocked", run_id)
            return {"status": "vetoed", "reason": f"blocked by {plan['binding_constraint']}", "plan": plan}
        _fill(ticker, "BUY", plan["shares"], price, "signal", signal_id)
        with db.engine().begin() as c:
            if held:
                total = held["shares"] + plan["shares"]
                avg = (held["shares"] * held["avg_price"] + plan["shares"] * price) / total
                c.execute(update(positions).where(positions.c.ticker == ticker).values(
                    shares=total, avg_price=avg, stop_loss=plan["stop_loss"], take_profit=plan["take_profit"], last_price=price))
            else:
                c.execute(insert(positions).values(ticker=ticker, shares=plan["shares"], avg_price=price,
                                                   stop_loss=plan["stop_loss"], take_profit=plan["take_profit"],
                                                   sector=sector, last_price=price, opened_at=db.now()))
        return {"status": "executed", "plan": plan}

    if action == "SELL":
        if not held:
            return {"status": "no_action", "reason": "SELL signal but no position held (long-only fund)"}
        close_position(held, price, "signal", signal_id)
        return {"status": "executed", "shares": held["shares"], "price": price}

    return {"status": "no_action", "reason": "HOLD"}


def close_position(pos: dict, price: float, reason: str, signal_id=None):
    pnl = (price - pos["avg_price"]) * pos["shares"]
    _fill(pos["ticker"], "SELL", pos["shares"], price, reason, signal_id, round(pnl, 2))
    with db.engine().begin() as c:
        c.execute(delete(positions).where(positions.c.ticker == pos["ticker"]))
    return pnl


def fill_pending(price_fn=None, max_age_days: int = 5) -> list:
    """Execute orders queued while the exchange was closed, at the current price with fresh sizing.
    Older orders expire; an order is superseded if a newer decision exists for the same ticker."""
    from datetime import timedelta

    from .db import signals
    from .tools import market_data as md
    from .tools import valuation as val

    price_fn = price_fn or md.latest_price
    done = []
    pending = db.fetch_all(select(signals).where(signals.c.status == "pending").order_by(signals.c.ts))
    for sig in pending:
        t, sid = sig["ticker"], sig["id"]
        newer = db.fetch_all(select(signals.c.id).where(signals.c.ticker == t, signals.c.ts > sig["ts"]))
        if newer:
            outcome = {"status": "superseded"}
        elif sig["ts"] < db.now() - timedelta(days=max_age_days):
            outcome = {"status": "expired"}
        else:
            try:
                px = float(price_fn(t))
                atr = val.atr(md.get_price_history(t, "3mo"))
                sector = (md.get_fundamentals(t) or {}).get("sector") or "Unknown"
            except Exception as exc:  # noqa: BLE001 - try again at the next check
                db.log_event("data_fallback", "fill_pending", f"{t}: {exc}")
                continue
            outcome = execute({"ticker": t, "action": sig["action"], "price": px, "confidence": sig["confidence"]},
                              sector, atr, signal_id=sid)
            plan = outcome.get("plan") or {}
            with db.engine().begin() as c:
                c.execute(update(signals).where(signals.c.id == sid).values(
                    price=px, stop_loss=plan.get("stop_loss", sig["stop_loss"]),
                    take_profit=plan.get("take_profit", sig["take_profit"]),
                    shares=plan.get("shares") or outcome.get("shares") or 0))
        with db.engine().begin() as c:
            c.execute(update(signals).where(signals.c.id == sid).values(status=outcome["status"]))
        if outcome["status"] == "executed":
            shares = (outcome.get("plan") or {}).get("shares") or outcome.get("shares")
            db.log_event("order_filled", t, f"{sig['action']} {shares} {t} at {px:,.2f} (queued at "
                                            f"{sig['ts']:%d %b %H:%M} UTC, filled at the market open)")
        else:
            db.log_event("order_cancelled", t, f"queued {sig['action']} {t} {outcome['status']}"
                                               + (f": {outcome.get('reason')}" if outcome.get("reason") else ""))
        done.append((t, sig["action"], outcome["status"]))
    return done


def mark_to_market(price_fn=None) -> dict:
    """Refresh prices, fire stop-loss / take-profit exits, record an equity snapshot.
    Needs no LLM, so the scheduler can run it every few minutes for free."""
    from .tools.market_data import latest_price

    price_fn = price_fn or latest_price
    exits = []
    for p in open_positions():
        try:
            px = price_fn(p["ticker"])
        except Exception as exc:  # noqa: BLE001
            db.log_event("data_fallback", "mark_to_market", f"{p['ticker']}: {exc}")
            continue
        if p["stop_loss"] and px <= p["stop_loss"]:
            exits.append((p["ticker"], "stop_loss", close_position(p, px, "stop_loss")))
        elif p["take_profit"] and px >= p["take_profit"]:
            exits.append((p["ticker"], "take_profit", close_position(p, px, "take_profit")))
        if exits and exits[-1][0] == p["ticker"]:
            _, why, pnl = exits[-1]
            level = p["stop_loss"] if why == "stop_loss" else p["take_profit"]
            db.log_event("auto_exit", why, f"Sold {p['shares']} {p['ticker']} at {px:,.2f}: "
                                           f"{why.replace('_', '-')} {level:,.2f} hit, P&L {pnl:+,.2f}")
        else:
            with db.engine().begin() as c:
                c.execute(update(positions).where(positions.c.ticker == p["ticker"]).values(last_price=px))
    try:
        bench = price_fn(config.M["benchmark"])
    except Exception:  # noqa: BLE001
        bench = None
    pos_value = sum(p["shares"] * (p["last_price"] or p["avg_price"]) for p in open_positions())
    snap = {"cash": round(cash(), 2), "positions_value": round(pos_value, 2),
            "equity": round(cash() + pos_value, 2), "benchmark": bench}
    with db.engine().begin() as c:
        c.execute(insert(snapshots).values(ts=db.now(), **snap))
    return {"snapshot": snap, "exits": exits}
