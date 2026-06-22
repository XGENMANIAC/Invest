"""
watch_loop.py — Deterministic watch engine for the trading watchdog.

BOUNDARY: This module NEVER calls any AI model. All language understanding
already happened once, in goal_parser.py. This loop only does math.
Execution of any trade stays entirely human-confirmed — this loop notifies only.

Dependencies: requests only (already installed). No pandas. No yfinance library.
Market data: Twelve Data (primary) → Finnhub (secondary) → Yahoo Finance (built-in fallback).
TD free tier: 8 req/min. Finnhub free tier: 60 req/min (crypto only on free).
Yahoo Finance: no key, no rate limit — always available as universal fallback.
"""

import math
import os
import sys
import time
import collections
import threading
from contextlib import nullcontext
from datetime import datetime, timezone

import requests

from notify import notify_event

# ── TWELVE DATA CONFIG ────────────────────────────────────────────────────────
TWELVE_BASE_URL   = "https://api.twelvedata.com"
TWELVE_OUTPUTSIZE = 50   # candles per request; 50 covers all indicators + buffer
TWELVE_RATE_LIMIT = int(os.environ.get("TWELVE_RATE_LIMIT", "8"))  # free: 8/min; paid: 55
TWELVE_API_KEY    = os.environ.get("TWELVE_API_KEY", "")

# ── FINNHUB CONFIG ────────────────────────────────────────────────────────────
FINNHUB_BASE_URL   = "https://finnhub.io/api/v1"
FINNHUB_RATE_LIMIT = int(os.environ.get("FINNHUB_RATE_LIMIT", "55"))  # free tier: 60/min
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")

# ── NON-BLOCKING RATE LIMITERS (per provider, shared across all watch threads) ─
# Each returns True and records the request if a slot is free; False otherwise.
# _fetch_candles tries TD first, falls over to FH instantly — never blocks.

_td_lock  = threading.Lock()
_td_times: collections.deque = collections.deque()

_fh_lock  = threading.Lock()
_fh_times: collections.deque = collections.deque()


def _td_rate_try() -> bool:
    with _td_lock:
        now = time.time()
        while _td_times and now - _td_times[0] >= 60.0:
            _td_times.popleft()
        if len(_td_times) < TWELVE_RATE_LIMIT:
            _td_times.append(now)
            return True
    return False


def _fh_rate_try() -> bool:
    with _fh_lock:
        now = time.time()
        while _fh_times and now - _fh_times[0] >= 60.0:
            _fh_times.popleft()
        if len(_fh_times) < FINNHUB_RATE_LIMIT:
            _fh_times.append(now)
            return True
    return False

# ── SYMBOL MAPS ───────────────────────────────────────────────────────────────
# Primary: Twelve Data symbols (what you type → TD ticker)
SYMBOL_TO_TICKER: dict[str, str] = {
    "GOLD":   "XAU/USD",
    "SILVER": "XAG/USD",
    "OIL":    "WTI/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "BNBUSD": "BNB/USD",
    "SOLUSD": "SOL/USD",
}

# Fallback: Finnhub symbols — (fh_symbol, endpoint_type)
# endpoint_type: "forex" | "crypto"  (Finnhub uses separate endpoints per asset class)
SYMBOL_TO_FH: dict[str, tuple[str, str]] = {
    "GOLD":   ("OANDA:XAU_USD",  "forex"),
    "SILVER": ("OANDA:XAG_USD",  "forex"),
    "OIL":    ("OANDA:BCO_USD",  "forex"),   # Brent crude; WTI not always available on OANDA
    "EURUSD": ("OANDA:EUR_USD",  "forex"),
    "GBPUSD": ("OANDA:GBP_USD",  "forex"),
    "USDJPY": ("OANDA:USD_JPY",  "forex"),
    "AUDUSD": ("OANDA:AUD_USD",  "forex"),
    "USDCAD": ("OANDA:USD_CAD",  "forex"),
    "USDCHF": ("OANDA:USD_CHF",  "forex"),
    "NZDUSD": ("OANDA:NZD_USD",  "forex"),
    "BTCUSD": ("BINANCE:BTCUSDT", "crypto"),
    "ETHUSD": ("BINANCE:ETHUSDT", "crypto"),
    "BNBUSD": ("BINANCE:BNBUSDT", "crypto"),
    "SOLUSD": ("BINANCE:SOLUSDT", "crypto"),
}

# Reverse map: Twelve Data ticker → Finnhub (symbol, type) for _fetch_candles dispatcher
_TD_TO_FH: dict[str, tuple[str, str]] = {
    SYMBOL_TO_TICKER[k]: SYMBOL_TO_FH[k]
    for k in SYMBOL_TO_TICKER
    if k in SYMBOL_TO_FH
}

# Twelve Data interval strings per rule timeframe
TIMEFRAME_TO_TWELVE: dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
}

# Finnhub resolution strings per Twelve Data interval string
_TD_INT_TO_FH_RES: dict[str, str] = {
    "1min": "1",
    "5min": "5",
    "15min": "15",
    "1h":   "60",
}

# ── YAHOO FINANCE MAPS (built-in fallback, no key required) ──────────────────
# Yahoo Finance v8 chart API — requires cookie session + crumb since late 2023.
_YF_CHART_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/",
    "https://query2.finance.yahoo.com/v8/finance/chart/",
]
_YF_COOKIE_SEED = "https://fc.yahoo.com"                               # seeds cookies
_YF_CRUMB_URL   = "https://query2.finance.yahoo.com/v1/test/getcrumb"  # returns crumb text
_YF_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Session state — initialised lazily, refreshed on 401/403
_yf_lock:    threading.Lock        = threading.Lock()
_yf_session: "requests.Session | None" = None
_yf_crumb:   str                   = ""
# Twelve Data ticker → Yahoo Finance ticker
_TD_TO_YF: dict[str, str] = {
    "XAU/USD": "GC=F",
    "XAG/USD": "SI=F",
    "WTI/USD": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "BNB/USD": "BNB-USD",
    "SOL/USD": "SOL-USD",
}
# Twelve Data interval string → (yf_interval, yf_range)
_TD_INT_TO_YF: dict[str, tuple[str, str]] = {
    "1min":  ("1m",  "1d"),
    "5min":  ("5m",  "5d"),
    "15min": ("15m", "5d"),
    "1h":    ("60m", "30d"),
}

# How many bars back to look for a price "touch" before checking close confirmation
TOUCH_LOOKBACK = 5  # bars back to search for a touch; 5 bars keeps it to ~25min on 5m

# Price within this multiple of tolerance triggers a one-time APPROACHING heads-up
APPROACHING_MULTIPLIER = 2.0

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
    clean = symbol.upper()
    # Strip Yahoo Finance-style suffixes that the LLM may hallucinate (=X, =F)
    for sfx in ("=X", "=F", "="):
        if clean.endswith(sfx):
            clean = clean[:-len(sfx)]
            break
    key = clean.replace("/", "").replace("-", "")
    if key in SYMBOL_TO_TICKER:
        return SYMBOL_TO_TICKER[key]
    if "/" in symbol:
        return symbol.upper()  # already a Twelve Data symbol like EUR/USD
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

# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def _td_ts(dt_str: str) -> float:
    """Parse a Twelve Data UTC datetime string to a Unix timestamp."""
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _fetch_from_twelve(ticker: str, interval: str) -> "Candles | None":
    """Fetch from Twelve Data. Caller must have already acquired a rate-limit slot."""
    try:
        resp = requests.get(
            f"{TWELVE_BASE_URL}/time_series",
            params={
                "symbol":     ticker,
                "interval":   interval,
                "outputsize": TWELVE_OUTPUTSIZE,
                "order":      "ASC",
                "timezone":   "UTC",
                "apikey":     TWELVE_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        msg = str(exc).replace(TWELVE_API_KEY, "<key>")
        _log(f"[TD] Fetch failed ({ticker}): {msg}")
        return None

    if data.get("status") == "error":
        _log(f"[TD] Error ({ticker}): {data.get('message', data)}")
        return None

    values = data.get("values")
    if not values:
        return None

    try:
        ts_, o_, h_, l_, c_, v_ = [], [], [], [], [], []
        for row in values:
            cv = _to_float(row.get("close"))
            if _nan(cv):
                continue
            ts_.append(_td_ts(row["datetime"]))
            o_.append(_to_float(row.get("open")))
            h_.append(_to_float(row.get("high")))
            l_.append(_to_float(row.get("low")))
            c_.append(cv)
            v_.append(_to_float(row.get("volume", 0)))
        return Candles(ts_, o_, h_, l_, c_, v_) if ts_ else None
    except (KeyError, TypeError, ValueError):
        return None


def _fetch_from_finnhub(fh_ticker: str, fh_type: str, resolution: str) -> "Candles | None":
    """
    Fetch from Finnhub candle endpoint. Caller must have already acquired a rate-limit slot.
    fh_type: 'forex' | 'crypto'  (selects the right Finnhub endpoint)
    resolution: Finnhub resolution string ('1', '5', '15', '60')
    """
    tf_min  = int(resolution) if resolution.isdigit() else 60
    now_ts  = int(time.time())
    # 7-day window ensures we get candles even after weekends/holidays
    from_ts = now_ts - 7 * 24 * 3600

    try:
        resp = requests.get(
            f"{FINNHUB_BASE_URL}/{fh_type}/candle",
            params={
                "symbol":     fh_ticker,
                "resolution": resolution,
                "from":       from_ts,
                "to":         now_ts,
                "token":      FINNHUB_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        msg = str(exc).replace(FINNHUB_API_KEY, "<key>")
        _log(f"[FH] Fetch failed ({fh_ticker}): {msg}")
        return None

    if data.get("s") != "ok":
        if data.get("s") == "no_data":
            _log(f"[FH] No data for {fh_ticker}")
        else:
            _log(f"[FH] Error ({fh_ticker}): {data}")
        return None

    try:
        rows = [
            (ts, ov, hv, lv, cv, vv)
            for ts, ov, hv, lv, cv, vv in zip(
                data.get("t", []),
                (_to_float(v) for v in data.get("o", [])),
                (_to_float(v) for v in data.get("h", [])),
                (_to_float(v) for v in data.get("l", [])),
                (_to_float(v) for v in data.get("c", [])),
                (_to_float(v) for v in data.get("v", [])),
            )
            if not _nan(cv)
        ]
        rows = rows[-TWELVE_OUTPUTSIZE:]  # keep most recent N candles
        if not rows:
            return None
        ts_, o_, h_, l_, c_, v_ = map(list, zip(*rows))
        return Candles(ts_, o_, h_, l_, c_, v_)
    except (KeyError, TypeError, ValueError):
        return None


def _yf_refresh_session() -> bool:
    """(Re)init Yahoo Finance session — seeds cookies then fetches crumb. Caller holds _yf_lock."""
    global _yf_session, _yf_crumb
    sess = requests.Session()
    sess.headers["User-Agent"] = _YF_UA
    try:
        sess.get(_YF_COOKIE_SEED, timeout=10)        # sets cookies
        r = sess.get(_YF_CRUMB_URL, timeout=10)
        if r.ok and r.text.strip():
            _yf_session, _yf_crumb = sess, r.text.strip()
            return True
    except Exception as exc:
        _log(f"[YF] Session init failed: {exc}")
    return False


def _fetch_from_yahoo(yf_ticker: str, yf_interval: str, yf_range: str) -> "Candles | None":
    """
    Fetch from Yahoo Finance v8 chart API (no API key required).
    Maintains a cookie+crumb session; refreshes it automatically on 401/403.
    """
    global _yf_session, _yf_crumb

    for attempt in range(2):        # attempt 0 = normal; attempt 1 = after session refresh
        with _yf_lock:
            if not _yf_session or not _yf_crumb:
                if not _yf_refresh_session():
                    return None
            sess, crumb = _yf_session, _yf_crumb

        for base in _YF_CHART_URLS:
            try:
                resp = sess.get(
                    f"{base}{yf_ticker}",
                    params={"interval": yf_interval, "range": yf_range, "crumb": crumb},
                    timeout=15,
                )
            except requests.RequestException as exc:
                _log(f"[YF] Network error ({yf_ticker}): {exc}")
                continue

            if resp.status_code in (401, 403) and attempt == 0:
                # Auth expired — invalidate and retry once
                with _yf_lock:
                    _yf_crumb = ""
                break   # break inner; outer loop retries with fresh session

            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                _log(f"[YF] Response error ({yf_ticker}): {exc}")
                continue

            try:
                chunk   = data["chart"]["result"][0]
                ts_list = chunk.get("timestamp") or []
                quote   = chunk["indicators"]["quote"][0]
                rows = [
                    (ts, _to_float(o), _to_float(h), _to_float(l), _to_float(c), _to_float(v))
                    for ts, o, h, l, c, v in zip(
                        ts_list,
                        quote.get("open",   []),
                        quote.get("high",   []),
                        quote.get("low",    []),
                        quote.get("close",  []),
                        quote.get("volume", []),
                    )
                    if ts is not None and not _nan(_to_float(c))
                ]
            except (KeyError, TypeError, IndexError):
                continue

            rows = rows[-TWELVE_OUTPUTSIZE:]
            if not rows:
                continue
            ts_, o_, h_, l_, c_, v_ = map(list, zip(*rows))
            return Candles(ts_, o_, h_, l_, c_, v_)

    return None


def _fetch_candles(ticker: str, interval: str) -> "Candles | None":
    """
    Fetch OHLCV candles with three-tier failover. Never blocks.

      1. Twelve Data  — fastest, rate-limited (8 req/min free)
      2. Finnhub      — free for crypto only; forex requires paid plan
      3. Yahoo Finance — no key, no rate limit, universal coverage

    ticker:   Twelve Data ticker (e.g. 'XAU/USD').
    interval: Twelve Data interval string (e.g. '15min').
    """
    # ── Twelve Data ──────────────────────────────────────────────────────────
    if TWELVE_API_KEY and _td_rate_try():
        result = _fetch_from_twelve(ticker, interval)
        if result is not None:
            return result

    # ── Finnhub fallback ─────────────────────────────────────────────────────
    fh_info = _TD_TO_FH.get(ticker)
    if fh_info and FINNHUB_API_KEY and _fh_rate_try():
        fh_ticker, fh_type = fh_info
        resolution = _TD_INT_TO_FH_RES.get(interval, "15")
        result = _fetch_from_finnhub(fh_ticker, fh_type, resolution)
        if result is not None:
            return result

    # ── Yahoo Finance built-in fallback ──────────────────────────────────────
    yf_ticker = _TD_TO_YF.get(ticker)
    if yf_ticker:
        yf_interval, yf_range = _TD_INT_TO_YF.get(interval, ("15m", "5d"))
        result = _fetch_from_yahoo(yf_ticker, yf_interval, yf_range)
        if result is not None:
            return result

    return None


def _is_fresh(candles: "Candles", tf: str) -> tuple[bool, float]:
    """
    Return (is_fresh: bool, age_minutes: float).
    Data is considered stale when the latest candle is older than max(10x the
    timeframe interval, 60 min). This catches weekend/market-closed freezes while
    tolerating normal intraday exchange pauses.
    """
    if not candles or not candles.timestamps:
        return False, float("inf")
    age_min = (time.time() - candles.timestamps[-1]) / 60
    tf_min  = _timeframe_minutes(tf)
    threshold = max(tf_min * 10, 60)  # at least 60 min; generous for exchange breaks
    return age_min <= threshold, age_min

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
        "candle_time":   candles.timestamps[-1] if candles.timestamps else None,
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
    Poll Twelve Data and evaluate the rule on every tick.

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

    if timeframe not in TIMEFRAME_TO_TWELVE:
        _log(f"Unknown timeframe {timeframe!r} — defaulting to 15m")
        timeframe = "15m"
    interval = TIMEFRAME_TO_TWELVE[timeframe]

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
                candles = _fetch_candles(ticker, interval)
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
    print("Watch Loop Self-Test  (Twelve Data)")
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
