"""
watch_loop.py — Deterministic watch engine for the trading watchdog.

BOUNDARY: This module NEVER calls any AI model. All language understanding
already happened once, in goal_parser.py. This loop only does math.
Execution of any trade stays entirely human-confirmed — this loop notifies only.

Dependencies: requests only (already installed). No pandas. No yfinance.
Market data is fetched directly from Yahoo Finance's public JSON API.
"""

import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import requests

from notify import notify_event

# ── SYMBOL → TICKER MAP ───────────────────────────────────────────────────────
# Edit here to add instruments. Key = what you type, value = Yahoo Finance ticker.
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

# Yahoo Finance API: (yf_interval, yf_range) per rule timeframe
TIMEFRAME_TO_YF: dict[str, tuple[str, str]] = {
    "1m":  ("1m",  "1d"),
    "5m":  ("5m",  "5d"),
    "15m": ("15m", "5d"),
    "1h":  ("60m", "30d"),
}

# How many bars back to look for a price "touch" before checking close confirmation
TOUCH_LOOKBACK = 10

# Price within this multiple of tolerance triggers a one-time APPROACHING heads-up
APPROACHING_MULTIPLIER = 2.0

# Fallback Yahoo Finance endpoints (tried in order)
_YF_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/",
    "https://query2.finance.yahoo.com/v8/finance/chart/",
]
_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ── CANDLE DATA STRUCTURE ─────────────────────────────────────────────────────

class Candles:
    """OHLCV data as plain Python lists. No pandas required."""
    __slots__ = ("timestamps", "open", "high", "low", "close", "volume")

    def __init__(self, timestamps, open_, high, low, close, volume):
        self.timestamps = timestamps
        self.open       = open_
        self.high       = high
        self.low        = low
        self.close      = close
        self.volume     = volume

    def __len__(self):
        return len(self.close)

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _nan(v) -> bool:
    """True if v is NaN or non-numeric."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_ticker(symbol: str) -> str:
    key = symbol.upper().replace("/", "")
    if key in SYMBOL_TO_TICKER:
        return SYMBOL_TO_TICKER[key]
    if any(ch in symbol for ch in ("=", "-")):
        return symbol  # already a valid Yahoo ticker
    raise ValueError(
        f"Unknown symbol {symbol!r}. Add it to SYMBOL_TO_TICKER in watch_loop.py."
    )


def _timeframe_minutes(tf: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(tf, 15)


def _keep_awake():
    """
    Optional keep-awake context manager via wakepy. Skips gracefully if not installed.

    NOTE: This loop only ticks while the device is awake. For overnight watches,
    keep the screen on, use a phone charging + Termux wake lock, or run on a VPS.
    """
    try:
        from wakepy import keep  # type: ignore
        _log("wakepy active — device will stay awake during watch.")
        return keep.running()
    except ImportError:
        print("[watch] Tip: pip install wakepy to prevent sleep on long watches.", file=sys.stderr)
        return nullcontext()
    except Exception as exc:
        print(f"[watch] wakepy unavailable ({exc}) — continuing without it.", file=sys.stderr)
        return nullcontext()

# ── DATA FETCHING (Yahoo Finance JSON API, no yfinance) ───────────────────────

def _fetch_candles(ticker: str, interval: str, yf_range: str) -> Candles | None:
    """
    Fetch OHLCV candles from Yahoo Finance's public v8 JSON API.
    Tries two endpoints for resilience. Returns Candles or None on any failure.
    """
    last_exc = None
    data = None

    for base in _YF_URLS:
        try:
            resp = requests.get(
                f"{base}{ticker}",
                params={"interval": interval, "range": yf_range},
                headers=_YF_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc

    if data is None:
        _log(f"Fetch failed ({ticker}): {last_exc}")
        return None

    try:
        result = data["chart"]["result"]
        if not result:
            _log(f"No chart data returned for {ticker}")
            return None

        r         = result[0]
        timestamps = r.get("timestamp") or []
        quote      = r["indicators"]["quote"][0]

        def _col(key):
            return [_to_float(v) for v in (quote.get(key) or [])]

        o, h, lo, c, v = _col("open"), _col("high"), _col("low"), _col("close"), _col("volume")

        # Drop rows where close is NaN (incomplete candles from current open bar)
        rows = [
            (ts, ov, hv, lv, cv, vv)
            for ts, ov, hv, lv, cv, vv in zip(timestamps, o, h, lo, c, v)
            if not _nan(cv)
        ]

        if not rows:
            _log(f"No valid candles for {ticker}")
            return None

        ts_, o_, h_, l_, c_, v_ = map(list, zip(*rows))
        return Candles(ts_, o_, h_, l_, c_, v_)

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _log(f"Parse error ({ticker}): {exc}")
        return None

# ── INDICATORS (pure Python, deterministic) ───────────────────────────────────

def _ema(values: list[float], period: int) -> list[float]:
    """Standard EMA with exponential smoothing (alpha = 2/(period+1))."""
    n      = len(values)
    result = [float("nan")] * n
    if n < period:
        return result
    alpha  = 2.0 / (period + 1)
    seed   = sum(values[:period]) / period
    result[period - 1] = seed
    ema = seed
    for i in range(period, n):
        ema = values[i] * alpha + ema * (1.0 - alpha)
        result[i] = ema
    return result


def _rsi(close: list[float], period: int = 14) -> list[float]:
    """Wilder RSI."""
    n      = len(close)
    result = [float("nan")] * n
    if n < period + 1:
        return result

    deltas = [close[i] - close[i - 1] for i in range(1, n)]

    avg_gain = sum(max(0.0, d) for d in deltas[:period]) / period
    avg_loss = sum(max(0.0, -d) for d in deltas[:period]) / period

    def _rs_to_rsi(ag, al):
        return 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    result[period] = _rs_to_rsi(avg_gain, avg_loss)

    for i in range(period, n - 1):
        d        = deltas[i]
        avg_gain = (avg_gain * (period - 1) + max(0.0, d))  / period
        avg_loss = (avg_loss * (period - 1) + max(0.0, -d)) / period
        result[i + 1] = _rs_to_rsi(avg_gain, avg_loss)

    return result


def _atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    """Wilder ATR."""
    n      = len(close)
    result = [float("nan")] * n
    if n < period + 1:
        return result

    tr = [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        for i in range(1, n)
    ]

    atr_val = sum(tr[:period]) / period
    result[period] = atr_val
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        result[i + 1] = atr_val

    return result


def _get_series(cond: dict, candles: Candles) -> list[float]:
    """Return the indicator or price series for a condition."""
    ind    = cond.get("indicator")
    params = cond.get("indicator_params") or {}
    if ind == "rsi":
        return _rsi(candles.close, params.get("period", 14))
    if ind == "ema_fast":
        return _ema(candles.close, params.get("period", 9))
    if ind == "ema_slow":
        return _ema(candles.close, params.get("period", 21))
    if ind == "atr":
        return _atr(candles.high, candles.low, candles.close, params.get("period", 14))
    return candles.close  # price conditions default to close

# ── CONDITION EVALUATION ──────────────────────────────────────────────────────

def _eval_one(cond: dict, candles: Candles, confirm_bars: int) -> tuple[bool, str]:
    """
    Evaluate a single condition against current candles.
    Returns (met: bool, human_note: str). Never raises.
    """
    op    = cond.get("operator", "")
    level = cond.get("level")
    tol   = cond.get("tolerance") or 0.0
    cid   = cond.get("id", "?")
    tf    = cond.get("timeframe", "15m")
    close = candles.close
    high  = candles.high
    low   = candles.low
    cur   = close[-1] if close else float("nan")
    n     = len(candles)

    if n < max(confirm_bars + 1, 3):
        return False, f"{cid}: not enough candles yet ({n})"

    # ── Retest operators (touch + close confirmation — never a bare wick) ─────
    if op == "touch_then_close_above":
        if level is None:
            return False, f"{cid}: level missing"
        lb      = min(TOUCH_LOOKBACK, n)
        touched = any(v <= level + tol for v in low[-lb:])
        if not touched:
            return False, f"{cid}: waiting for touch of {level:.5f} (cur {cur:.5f})"
        ok = all(v > level for v in close[-confirm_bars:])
        return ok, (f"{cid}: touched {level:.5f}, confirmed above"
                    if ok else f"{cid}: touched {level:.5f}, awaiting close above")

    if op == "touch_then_close_below":
        if level is None:
            return False, f"{cid}: level missing"
        lb      = min(TOUCH_LOOKBACK, n)
        touched = any(v >= level - tol for v in high[-lb:])
        if not touched:
            return False, f"{cid}: waiting for touch of {level:.5f} (cur {cur:.5f})"
        ok = all(v < level for v in close[-confirm_bars:])
        return ok, (f"{cid}: touched {level:.5f}, confirmed below"
                    if ok else f"{cid}: touched {level:.5f}, awaiting close below")

    # ── Breakout operators (close beyond level, not a wick) ───────────────────
    if op == "close_above":
        if level is None:
            return False, f"{cid}: level missing"
        ok = all(v > level for v in close[-confirm_bars:])
        return ok, f"{cid}: {'broke above' if ok else 'below'} {level:.5f} (cur {cur:.5f})"

    if op == "close_below":
        if level is None:
            return False, f"{cid}: level missing"
        ok = all(v < level for v in close[-confirm_bars:])
        return ok, f"{cid}: {'broke below' if ok else 'above'} {level:.5f} (cur {cur:.5f})"

    # ── Indicator / cross operators ───────────────────────────────────────────
    if op == "crosses_above":
        series = [v for v in _get_series(cond, candles) if not _nan(v)]
        if len(series) < 2:
            return False, f"{cid}: insufficient data for cross check"
        ok = series[-2] <= level < series[-1]
        return ok, f"{cid}: {'crossed above' if ok else 'no cross'} {level}"

    if op == "crosses_below":
        series = [v for v in _get_series(cond, candles) if not _nan(v)]
        if len(series) < 2:
            return False, f"{cid}: insufficient data for cross check"
        ok = series[-2] >= level > series[-1]
        return ok, f"{cid}: {'crossed below' if ok else 'no cross'} {level}"

    if op == "gte":
        series = _get_series(cond, candles)
        val    = series[-1]
        ok     = not _nan(val) and val >= level
        return ok, f"{cid}: {val:.4f} {'≥' if ok else '<'} {level}"

    if op == "lte":
        series = _get_series(cond, candles)
        val    = series[-1]
        ok     = not _nan(val) and val <= level
        return ok, f"{cid}: {val:.4f} {'≤' if ok else '>'} {level}"

    # ── Range / stall operator ────────────────────────────────────────────────
    if op == "stalls":
        tf_mins  = _timeframe_minutes(tf)
        lookback = min(max(4, 60 // tf_mins), n)
        recent   = close[-lookback:]
        if len(recent) < 2:
            return False, f"{cid}: not enough data for stall check"
        price_range = max(recent) - min(recent)
        ok = price_range <= tol
        return ok, (f"{cid}: range {price_range:.5f} {'≤' if ok else '>'} "
                    f"tol {tol:.5f} over {lookback} bars")

    if op == "expires":
        return False, f"{cid}: timeout handled by run_watch"

    return False, f"{cid}: unknown operator {op!r}"


def evaluate_conditions(rule: dict, candles: Candles) -> dict:
    """
    Pure function — evaluates all conditions against current candles.

    Returns:
        {"triggered": bool, "which": [ids met], "detail": str, "current_price": float}
    Never raises.
    """
    conditions   = rule.get("conditions", [])
    logic        = rule.get("logic", "ALL")
    confirm_bars = max(1, int(rule.get("confirm_bars", 1)))
    current_price = float(candles.close[-1])

    active = [c for c in conditions if c.get("operator") != "expires"]

    met_ids, notes = [], []
    for cond in active:
        try:
            ok, note = _eval_one(cond, candles, confirm_bars)
        except Exception as exc:
            ok, note = False, f"{cond.get('id','?')}: eval error — {exc}"
        notes.append(note)
        if ok:
            met_ids.append(cond.get("id", "?"))

    if logic == "ALL":
        triggered = len(active) > 0 and len(met_ids) == len(active)
    else:
        triggered = len(met_ids) > 0

    return {
        "triggered":     triggered,
        "which":         met_ids,
        "detail":        " | ".join(notes) if notes else "no active conditions",
        "current_price": current_price,
    }

# ── APPROACHING HEADS-UP ──────────────────────────────────────────────────────

def _check_approaching(rule: dict, candles: Candles, heads_up_fired: set) -> None:
    """Fire a one-time APPROACHING event when price is within 2× tolerance of a level."""
    symbol = rule["symbol"]
    cur    = float(candles.close[-1])

    for cond in rule.get("conditions", []):
        if cond.get("type") != "price_level":
            continue
        cid   = cond.get("id", "?")
        level = cond.get("level")
        tol   = cond.get("tolerance") or 0.0
        if level is None or cid in heads_up_fired:
            continue
        if abs(cur - level) <= tol * APPROACHING_MULTIPLIER:
            notify_event(
                "APPROACHING", symbol,
                f"price nearing {level:.5f} — {cond.get('notes', '')}",
                current_price=cur,
            )
            heads_up_fired.add(cid)
            _log(f"Approaching alert fired: {cid} (level {level})")

# ── WATCH LOOP ────────────────────────────────────────────────────────────────

def run_watch(
    rule:          dict,
    poll_seconds:  int  = 20,
    one_shot:      bool = True,
    heads_up:      bool = True,
    on_event:      object = None,
) -> None:
    """
    Poll Yahoo Finance and evaluate the rule on every tick.

    Args:
        rule:          Structured rule dict from goal_parser.parse_goal().
        poll_seconds:  Seconds between ticks (default 20).
        one_shot:      Stop after the first trigger (default True). Set False for
                       trade-management goals that need continuous monitoring.
        heads_up:      Fire one APPROACHING event when price nears a level (default True).
        on_event:      Optional callback(result_dict) when a trigger or timeout fires.

    Never raises. Stops on trigger, timeout, or Ctrl+C.
    """
    symbol     = rule.get("symbol", "?")
    timeframe  = rule.get("timeframe", "15m")
    window_m   = rule.get("window_minutes")
    timeout_ev = rule.get("on_timeout_event", "TIMEOUT")

    try:
        ticker = _resolve_ticker(symbol)
    except ValueError as exc:
        _log(f"Cannot start: {exc}")
        return

    if timeframe not in TIMEFRAME_TO_YF:
        _log(f"Unknown timeframe {timeframe!r} — defaulting to 15m")
        timeframe = "15m"
    interval, yf_range = TIMEFRAME_TO_YF[timeframe]

    start    = time.time()
    deadline = start + window_m * 60 if window_m else None

    heads_up_fired: set = set()
    tick = 0

    _log(
        f"Watch started: {symbol} ({ticker}) | {timeframe} | poll={poll_seconds}s"
        f" | window={window_m}min | {rule.get('human_summary', '')}"
    )
    _log(f"Fires {rule.get('event_type')} on trigger | one_shot={one_shot}")

    with _keep_awake():
        while True:
            tick += 1
            now         = time.time()
            elapsed_min = (now - start) / 60

            # ── Timeout ───────────────────────────────────────────────────
            if deadline and now >= deadline:
                _log(f"Tick {tick} | Deadline reached ({elapsed_min:.1f}min) → {timeout_ev}")
                if timeout_ev:
                    notify_event(
                        timeout_ev, symbol,
                        f"watch window expired after {elapsed_min:.0f}min with no trigger",
                    )
                if on_event:
                    on_event({"trigger": "timeout", "event_type": timeout_ev})
                break

            # ── Fetch ─────────────────────────────────────────────────────
            try:
                candles = _fetch_candles(ticker, interval, yf_range)
            except Exception as exc:
                _log(f"Tick {tick} | Unexpected error during fetch: {exc} — skipping")
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

            cur     = result["current_price"]
            trig    = result["triggered"]
            detail  = result["detail"]
            tl      = f"{(deadline - now)/60:.0f}min left" if deadline else "open-ended"

            _log(
                f"Tick {tick} | {symbol} @ {cur:.5f} | {elapsed_min:.1f}min elapsed"
                f" ({tl}) | {'*** TRIGGERED ***' if trig else 'watching'}"
                f" | {detail}"
            )

            # ── Approaching heads-up ──────────────────────────────────────
            if heads_up and not trig:
                try:
                    _check_approaching(rule, candles, heads_up_fired)
                except Exception as exc:
                    _log(f"Approaching check error: {exc}")

            # ── Trigger ───────────────────────────────────────────────────
            if trig:
                event_type = rule.get("event_type", "INFO")
                notify_event(
                    event_type, symbol,
                    rule.get("human_summary", detail),
                    current_price=cur,
                )
                _log(f"Alert sent: {event_type} for {symbol} @ {cur:.5f}")
                if on_event:
                    on_event({**result, "event_type": event_type})
                if one_shot:
                    _log("one_shot=True — stopping after trigger.")
                    break

            # ── Sleep (shorten if deadline is near) ───────────────────────
            sleep_secs = poll_seconds
            if deadline:
                remaining  = deadline - time.time()
                sleep_secs = min(poll_seconds, max(1.0, remaining))
            try:
                time.sleep(sleep_secs)
            except KeyboardInterrupt:
                raise

    _log("Watch loop ended.")

# ── CONVENIENCE: PARSE + WATCH IN ONE CALL ────────────────────────────────────

def watch_from_goal(
    goal_text:     str,
    symbol:        str,
    current_price: float | None = None,
    levels:        dict  | None = None,
    poll_seconds:  int          = 20,
    one_shot:      bool         = True,
    heads_up:      bool         = True,
) -> dict | None:
    """
    Parse a natural-language goal and immediately start watching.
    Returns the parsed rule on success, None if clarification is needed.
    """
    from goal_parser import parse_goal

    _log(f"Parsing: {goal_text!r}")
    rule = parse_goal(goal_text, symbol, current_price=current_price, levels=levels)

    if rule.get("status") == "needs_clarification":
        print(f"\nNeeds clarification: {rule.get('clarification')}\n")
        return None

    _log(f"Rule: {rule.get('human_summary')}")
    if rule.get("assumptions"):
        _log(f"Assumptions: {rule['assumptions']}")

    try:
        run_watch(rule, poll_seconds=poll_seconds, one_shot=one_shot, heads_up=heads_up)
    except KeyboardInterrupt:
        _log("Stopped by user.")

    return rule

# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Inline rule — no parser call needed. Adjust `level` to be near the
    # actual USDJPY price to see conditions fire sooner.
    sample_rule = {
        "status":           "ok",
        "symbol":           "USDJPY",
        "timeframe":        "5m",
        "event_type":       "ENTRY_TRIGGER",
        "human_summary":    "USDJPY retest 160.50 and hold for 2 bars",
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
                "notes":            "retest 160.50 support, 2-bar close above",
            }
        ],
        "logic":            "ALL",
        "confirm_bars":     2,
        "window_minutes":   60,
        "on_timeout_event": "TIMEOUT",
        "assumptions":      ["tolerance 0.08 = 0.05% of 160.50"],
    }

    print("=" * 60)
    print("Watch Loop Self-Test  (no pandas, no yfinance)")
    print(f"Symbol:  {sample_rule['symbol']}")
    print(f"Level:   {sample_rule['conditions'][0]['level']}")
    print(f"Poll:    15s  |  Window: {sample_rule['window_minutes']}min")
    print("Tip: change `level` to the current USDJPY price to see")
    print("     a trigger fire immediately.")
    print("Press Ctrl+C to stop early.")
    print("=" * 60)

    try:
        run_watch(sample_rule, poll_seconds=15, one_shot=True, heads_up=True)
    except KeyboardInterrupt:
        _log("Stopped.")
