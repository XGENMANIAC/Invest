"""
watch_loop.py — Deterministic watch engine for the trading watchdog.

BOUNDARY: This module NEVER calls any AI model. All language understanding
already happened once, in goal_parser.py. This loop only does math.
Execution of any trade stays entirely human-confirmed — this loop notifies only.
"""

import sys
import time
from contextlib import nullcontext
from datetime import datetime

import pandas as pd
import yfinance as yf

from notify import notify_event

# ── SYMBOL → TICKER MAP ───────────────────────────────────────────────────────
# Key = what you or the parser uses; value = yfinance ticker.
# Add new instruments here — one line per market.
SYMBOL_TO_TICKER: dict[str, str] = {
    "GOLD":   "GC=F",
    "SILVER": "SI=F",
    "OIL":    "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
    "SOLUSD": "SOL-USD",
}

# yfinance (interval, period) pairs per rule timeframe
TIMEFRAME_TO_YF: dict[str, tuple[str, str]] = {
    "1m":  ("1m",  "1d"),
    "5m":  ("5m",  "1d"),
    "15m": ("15m", "1d"),
    "1h":  ("60m", "5d"),
}

# How many bars back to search for a price "touch" before requiring a close confirmation.
# Keep small to avoid triggering on stale historical touches.
TOUCH_LOOKBACK = 10

# Price must be within this multiple of tolerance to fire an APPROACHING heads-up.
APPROACHING_MULTIPLIER = 2.0

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _resolve_ticker(symbol: str) -> str:
    """Map a user symbol to a yfinance ticker. Raises ValueError if unknown."""
    key = symbol.upper().replace("/", "")
    if key in SYMBOL_TO_TICKER:
        return SYMBOL_TO_TICKER[key]
    # Pass-through if the symbol already looks like a yfinance ticker
    if any(ch in symbol for ch in ("=", "-")):
        return symbol
    raise ValueError(
        f"Unknown symbol {symbol!r}. Add it to SYMBOL_TO_TICKER in watch_loop.py."
    )


def _timeframe_minutes(tf: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(tf, 15)


def _keep_awake():
    """
    Return a context manager that keeps the machine awake while watching.
    Requires the optional `wakepy` package; gracefully skips if not installed.

    NOTE: This loop only ticks while the machine is awake. For overnight or
    long watches, either use wakepy, run on an always-on box (VPS/Termux on
    a plugged-in phone), or keep the screen/lid active manually.
    """
    try:
        from wakepy import keep  # type: ignore
        _log("wakepy active — machine will stay awake for this watch.")
        return keep.running()
    except ImportError:
        print(
            "[watch] Tip: pip install wakepy to prevent sleep on long watches.",
            file=sys.stderr,
        )
        return nullcontext()
    except Exception as exc:
        print(f"[watch] wakepy unavailable ({exc}) — continuing without it.", file=sys.stderr)
        return nullcontext()

# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def _fetch_candles(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles from yfinance. Returns a clean DataFrame or None on failure.
    A None return means "skip this tick" — the loop continues.
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            _log(f"yfinance returned empty frame for {ticker} ({interval}/{period})")
            return None
        # Flatten multi-index columns yfinance sometimes produces
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if not df.empty else None
    except Exception as exc:
        _log(f"Fetch error ({ticker}): {exc}")
        return None

# ── INDICATORS ────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _indicator_series(cond: dict, candles: pd.DataFrame) -> pd.Series:
    """Return the indicator or price series for a condition."""
    ind    = cond.get("indicator")
    params = cond.get("indicator_params") or {}
    close  = candles["Close"]
    if ind == "rsi":
        return _rsi(close, params.get("period", 14))
    if ind == "ema_fast":
        return _ema(close, params.get("period", 9))
    if ind == "ema_slow":
        return _ema(close, params.get("period", 21))
    if ind == "atr":
        return _atr(candles["High"], candles["Low"], close, params.get("period", 14))
    return close  # price conditions fall back to close

# ── CONDITION EVALUATION ──────────────────────────────────────────────────────

def _eval_one(cond: dict, candles: pd.DataFrame, confirm_bars: int) -> tuple[bool, str]:
    """
    Evaluate a single condition. Returns (met: bool, human_note: str).
    All operator semantics are defined here — the single source of truth for what each
    operator means when evaluated against OHLCV data.
    """
    op    = cond.get("operator", "")
    level = cond.get("level")
    tol   = cond.get("tolerance") or 0
    cid   = cond.get("id", "?")
    tf    = cond.get("timeframe", "15m")
    close = candles["Close"]
    high  = candles["High"]
    low   = candles["Low"]
    cur   = close.iloc[-1]

    # Require enough bars for confirm_bars checks
    if len(candles) < max(confirm_bars + 1, 3):
        return False, f"{cid}: not enough candles yet"

    # ── Price level operators ──────────────────────────────────────────────
    if op == "touch_then_close_above":
        # Retest of support: price tagged the level (low within tolerance), then confirmed
        # by closing above it for confirm_bars consecutive candles.
        if level is None:
            return False, f"{cid}: level missing"
        touched = (candles["Low"].tail(TOUCH_LOOKBACK) <= level + tol).any()
        if not touched:
            return False, f"{cid}: waiting for touch of {level:.5f} (cur {cur:.5f})"
        confirmed = (close.iloc[-confirm_bars:] > level).all()
        return (confirmed,
                f"{cid}: touched {level:.5f}, {'confirmed above' if confirmed else 'awaiting close above'}")

    if op == "touch_then_close_below":
        # Retest of resistance: price tagged the level (high within tolerance), then
        # closed below it for confirm_bars consecutive candles.
        if level is None:
            return False, f"{cid}: level missing"
        touched = (candles["High"].tail(TOUCH_LOOKBACK) >= level - tol).any()
        if not touched:
            return False, f"{cid}: waiting for touch of {level:.5f} (cur {cur:.5f})"
        confirmed = (close.iloc[-confirm_bars:] < level).all()
        return (confirmed,
                f"{cid}: touched {level:.5f}, {'confirmed below' if confirmed else 'awaiting close below'}")

    if op == "close_above":
        # Upside breakout: last confirm_bars candles all closed above level (not a wick).
        if level is None:
            return False, f"{cid}: level missing"
        met = (close.iloc[-confirm_bars:] > level).all()
        return met, f"{cid}: {'broke above' if met else 'below'} {level:.5f} (cur {cur:.5f})"

    if op == "close_below":
        # Downside breakdown: last confirm_bars candles all closed below level.
        if level is None:
            return False, f"{cid}: level missing"
        met = (close.iloc[-confirm_bars:] < level).all()
        return met, f"{cid}: {'broke below' if met else 'above'} {level:.5f} (cur {cur:.5f})"

    # ── Indicator / cross operators ────────────────────────────────────────
    if op == "crosses_above":
        series = _indicator_series(cond, candles).dropna()
        if len(series) < 2:
            return False, f"{cid}: insufficient data for cross"
        met = series.iloc[-2] <= level < series.iloc[-1]
        return met, f"{cid}: {'crossed above' if met else 'no cross'} {level}"

    if op == "crosses_below":
        series = _indicator_series(cond, candles).dropna()
        if len(series) < 2:
            return False, f"{cid}: insufficient data for cross"
        met = series.iloc[-2] >= level > series.iloc[-1]
        return met, f"{cid}: {'crossed below' if met else 'no cross'} {level}"

    if op == "gte":
        series = _indicator_series(cond, candles)
        val = series.iloc[-1]
        met = not pd.isna(val) and val >= level
        return met, f"{cid}: {val:.4f} {'≥' if met else '<'} {level}"

    if op == "lte":
        series = _indicator_series(cond, candles)
        val = series.iloc[-1]
        met = not pd.isna(val) and val <= level
        return met, f"{cid}: {val:.4f} {'≤' if met else '>'} {level}"

    # ── Range / stall operator ─────────────────────────────────────────────
    if op == "stalls":
        # Price has moved less than tolerance over the last ~1h of bars.
        # lookback ≈ 1 hour worth of candles on the condition's timeframe.
        tf_mins  = _timeframe_minutes(tf)
        lookback = max(4, 60 // tf_mins)
        recent   = close.tail(lookback)
        if len(recent) < 2:
            return False, f"{cid}: not enough data for stall check"
        price_range = recent.max() - recent.min()
        met = price_range <= tol
        return (met,
                f"{cid}: range {price_range:.5f} {'≤' if met else '>'} tol {tol:.5f} "
                f"over {lookback} bars")

    if op == "expires":
        # Handled by the timeout logic in run_watch, not here.
        return False, f"{cid}: timeout handled by run_watch"

    return False, f"{cid}: unrecognised operator {op!r}"


def evaluate_conditions(rule: dict, candles: pd.DataFrame) -> dict:
    """
    Pure function. Evaluates all rule conditions against current candles.

    Returns:
        {
            "triggered": bool,
            "which": [condition ids that are currently met],
            "detail": "human-readable summary",
            "current_price": float,
        }
    Never raises.
    """
    conditions   = rule.get("conditions", [])
    logic        = rule.get("logic", "ALL")
    confirm_bars = max(1, int(rule.get("confirm_bars", 1)))
    current_price = float(candles["Close"].iloc[-1])

    active = [c for c in conditions if c.get("operator") != "expires"]

    met_ids, notes = [], []
    for cond in active:
        try:
            result, note = _eval_one(cond, candles, confirm_bars)
        except Exception as exc:
            result, note = False, f"{cond.get('id', '?')}: eval error — {exc}"
        notes.append(note)
        if result:
            met_ids.append(cond.get("id", "?"))

    if logic == "ALL":
        triggered = len(met_ids) == len(active) and len(active) > 0
    else:  # ANY
        triggered = len(met_ids) > 0

    return {
        "triggered":     triggered,
        "which":         met_ids,
        "detail":        " | ".join(notes) if notes else "no active conditions",
        "current_price": current_price,
    }

# ── APPROACHING HEADS-UP ──────────────────────────────────────────────────────

def _check_approaching(
    rule: dict, candles: pd.DataFrame, heads_up_fired: set
) -> None:
    """
    Fire a single APPROACHING event per price-level condition when price moves
    within APPROACHING_MULTIPLIER × tolerance of the level but hasn't confirmed.
    Deduped via heads_up_fired so it fires at most once per condition per watch.
    """
    symbol = rule["symbol"]
    current_price = float(candles["Close"].iloc[-1])

    for cond in rule.get("conditions", []):
        if cond.get("type") != "price_level":
            continue
        cid   = cond.get("id", "?")
        level = cond.get("level")
        tol   = cond.get("tolerance") or 0
        if level is None or cid in heads_up_fired:
            continue
        if abs(current_price - level) <= tol * APPROACHING_MULTIPLIER:
            notify_event(
                "APPROACHING", symbol,
                f"price near {level:.5f} — {cond.get('notes', '')}",
                current_price=current_price,
            )
            heads_up_fired.add(cid)
            _log(f"Approaching alert fired: condition {cid} (level {level})")

# ── WATCH LOOP ────────────────────────────────────────────────────────────────

def run_watch(
    rule:         dict,
    poll_seconds: int  = 20,
    one_shot:     bool = True,
    heads_up:     bool = True,
    on_event:     object = None,
) -> None:
    """
    Poll for candles and evaluate the rule on each tick.

    Args:
        rule:         Structured rule dict from goal_parser.parse_goal().
        poll_seconds: Seconds between ticks (default 20).
        one_shot:     Stop after the first trigger fires (default True).
                      Set False for trade-management goals that need continuous monitoring.
        heads_up:     Fire an APPROACHING event when price nears a level (default True).
        on_event:     Optional callback(result_dict) called when a trigger or timeout fires.

    Never raises. Runs until triggered, timeout, or KeyboardInterrupt.
    """
    symbol   = rule.get("symbol", "?")
    timeframe = rule.get("timeframe", "15m")
    window_m  = rule.get("window_minutes")
    timeout_ev = rule.get("on_timeout_event", "TIMEOUT")

    try:
        ticker = _resolve_ticker(symbol)
    except ValueError as exc:
        _log(f"Cannot start: {exc}")
        return

    if timeframe not in TIMEFRAME_TO_YF:
        _log(f"Unknown timeframe {timeframe!r} — defaulting to 15m")
        timeframe = "15m"
    interval, period = TIMEFRAME_TO_YF[timeframe]

    start    = time.time()
    deadline = start + window_m * 60 if window_m else None

    heads_up_fired: set = set()
    tick = 0

    _log(
        f"Watch started: {symbol} ({ticker}) | {timeframe} | "
        f"poll={poll_seconds}s | window={window_m}min | "
        f"goal: {rule.get('human_summary', 'n/a')}"
    )
    _log(f"Event type on trigger: {rule.get('event_type')} | one_shot={one_shot}")

    with _keep_awake():
        while True:
            tick += 1
            now         = time.time()
            elapsed_min = (now - start) / 60

            # ── Timeout check ────────────────────────────────────────────
            if deadline and now >= deadline:
                _log(f"Tick {tick} | Window expired ({elapsed_min:.1f}min) → {timeout_ev}")
                if timeout_ev:
                    notify_event(
                        timeout_ev, symbol,
                        f"watch window expired after {elapsed_min:.0f}min, no trigger",
                    )
                if on_event:
                    on_event({"trigger": "timeout", "event_type": timeout_ev,
                              "elapsed_min": elapsed_min})
                break

            # ── Fetch ─────────────────────────────────────────────────────
            try:
                candles = _fetch_candles(ticker, interval, period)
            except Exception as exc:
                _log(f"Tick {tick} | Unexpected fetch error: {exc} — skipping")
                candles = None

            if candles is None:
                _log(f"Tick {tick} | No data — will retry next tick")
                time.sleep(poll_seconds)
                continue

            # ── Evaluate ──────────────────────────────────────────────────
            try:
                result = evaluate_conditions(rule, candles)
            except Exception as exc:
                _log(f"Tick {tick} | Condition eval error: {exc} — skipping")
                time.sleep(poll_seconds)
                continue

            cur   = result["current_price"]
            trig  = result["triggered"]
            detail = result["detail"]
            time_left = f"{(deadline - now) / 60:.0f}min left" if deadline else "no deadline"

            _log(
                f"Tick {tick} | {symbol} @ {cur:.5f} | {elapsed_min:.1f}min elapsed "
                f"({time_left}) | {'*** TRIGGERED ***' if trig else 'watching'} | {detail}"
            )

            # ── Approaching heads-up (only when not yet triggered) ────────
            if heads_up and not trig:
                try:
                    _check_approaching(rule, candles, heads_up_fired)
                except Exception as exc:
                    _log(f"Approaching check error: {exc}")

            # ── Trigger ───────────────────────────────────────────────────
            if trig:
                event_type = rule.get("event_type", "INFO")
                message    = rule.get("human_summary", detail)
                notify_event(event_type, symbol, message, current_price=cur)
                _log(f"Alert sent: {event_type} for {symbol} @ {cur:.5f}")
                if on_event:
                    on_event({**result, "event_type": event_type})
                if one_shot:
                    _log("one_shot=True — stopping after first trigger.")
                    break

            # ── Sleep (shorten sleep if close to deadline) ─────────────
            sleep_secs = poll_seconds
            if deadline:
                remaining = deadline - time.time()
                sleep_secs = min(poll_seconds, max(1, remaining))
            try:
                time.sleep(sleep_secs)
            except KeyboardInterrupt:
                raise  # let the outer handler catch it

    _log("Watch loop ended.")

# ── CONVENIENCE: GOAL → WATCH IN ONE CALL ─────────────────────────────────────

def watch_from_goal(
    goal_text:    str,
    symbol:       str,
    current_price: float | None = None,
    levels:        dict  | None = None,
    poll_seconds:  int         = 20,
    one_shot:      bool        = True,
    heads_up:      bool        = True,
) -> dict | None:
    """
    Parse a natural-language goal and immediately start watching.
    This is the one-call interface the app uses.

    Returns the parsed rule on success, None if parsing failed or needs clarification.
    """
    from goal_parser import parse_goal

    _log(f"Parsing goal: {goal_text!r}")
    rule = parse_goal(goal_text, symbol, current_price=current_price, levels=levels)

    if rule.get("status") == "needs_clarification":
        print(f"\nClarification needed: {rule.get('clarification')}\n")
        return None

    _log(f"Rule ready: {rule.get('human_summary')}")
    _log(f"Assumptions: {rule.get('assumptions', [])}")

    try:
        run_watch(rule, poll_seconds=poll_seconds, one_shot=one_shot, heads_up=heads_up)
    except KeyboardInterrupt:
        _log("Stopped by user.")

    return rule

# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Inline rule — no parser call, so this tests the loop in isolation.
    # Adjust `level` to be near current USDJPY price so you can see conditions fire.
    sample_rule = {
        "status":         "ok",
        "symbol":         "USDJPY",
        "timeframe":      "5m",
        "event_type":     "ENTRY_TRIGGER",
        "human_summary":  "USDJPY retest 160.50 and hold",
        "conditions": [
            {
                "id":               "c1",
                "type":             "price_level",
                "operator":         "touch_then_close_above",
                "level":            160.50,
                "tolerance":        0.08,
                "indicator":        None,
                "indicator_params": None,
                "timeframe":        "5m",
                "notes":            "price retests 160.50 support and closes above for 2 bars",
            }
        ],
        "logic":            "ALL",
        "confirm_bars":     2,
        "window_minutes":   60,
        "on_timeout_event": "TIMEOUT",
        "assumptions":      ["tolerance defaulted to 0.05% of level", "15m default overridden to 5m by goal"],
    }

    print("=" * 60)
    print("Watch Loop Self-Test")
    print(f"Symbol : {sample_rule['symbol']}")
    print(f"Level  : {sample_rule['conditions'][0]['level']}")
    print(f"Poll   : 15s  |  Window: {sample_rule['window_minutes']}min")
    print("Tip    : change `level` above to the current USDJPY price to")
    print("         see a trigger fire immediately.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        run_watch(sample_rule, poll_seconds=15, one_shot=True, heads_up=True)
    except KeyboardInterrupt:
        _log("Stopped.")
