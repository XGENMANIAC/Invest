"""
goal_parser.py — Natural-language goal → deterministic watch rule, via Kimi K2.6 on NVIDIA NIM.

BOUNDARY: This module ONLY translates a goal into a rule.
It does NOT watch prices, send notifications, or place trades.
Call parse_goal() once when a goal is set; hand the returned dict to the watch loop.
"""

import json
import os
import sys
import time
import warnings

import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────
# To use a self-hosted NIM container, change NIM_BASE_URL to e.g. "http://localhost:8000/v1"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Confirm the exact model ID in the NIM catalog: https://integrate.api.nvidia.com/
# Swap this one line to target a different NIM-served model.
NIM_MODEL = "moonshotai/kimi-k2.6"

NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
if not NIM_API_KEY:
    warnings.warn(
        "[goal_parser] NIM_API_KEY env var is not set. parse_goal() will return "
        "needs_clarification until it is set.",
        stacklevel=2,
    )

REQUEST_TIMEOUT = 60  # seconds; structured output from Kimi can take a moment

# ── ALLOWED VALUES (must stay in sync with notify.py EventType) ───────────────
ALLOWED_EVENT_TYPES = {
    "ENTRY_TRIGGER", "APPROACHING", "INVALIDATED", "MANAGE_TRADE",
    "TARGET_HIT", "STOP_RISK", "TIMEOUT", "RANGE_STALL", "INDICATOR_STATE", "INFO",
}
ALLOWED_TIMEFRAMES = {"1m", "5m", "15m", "1h"}
REQUIRED_RULE_KEYS = {
    "status", "symbol", "timeframe", "event_type",
    "conditions", "logic", "confirm_bars",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# This is the heart of the module. Edit here to change parsing behaviour.
_SYSTEM_PROMPT = """You are a trading-goal-to-rule compiler. Your ONLY output is a single JSON object — no prose, no explanation, no markdown, no code fences.

You receive a natural-language watch goal and optional context (symbol, current price, key levels).
Translate the goal into a fully explicit, deterministic watch rule that a Python tick-checker can evaluate against OHLCV candle data without any further interpretation. Every field must be a concrete number or a string from the allowed lists — never a vague description where a number is expected.

════════════════════════════════════════
OUTPUT SCHEMA — return EXACTLY this shape
════════════════════════════════════════
{
  "status": "ok" | "needs_clarification",
  "clarification": "<one-line question if status=needs_clarification, else null>",
  "symbol": "<echo the SYMBOL from the input exactly as given, e.g. USDJPY>",
  "timeframe": "1m" | "5m" | "15m" | "1h",
  "event_type": "<one of the EXACT strings listed below>",
  "human_summary": "<one short line restating the goal in plain words>",
  "conditions": [
    {
      "id": "c1",
      "type": "price_level" | "indicator" | "time" | "structure",
      "operator": "<one of the EXACT operator strings listed below>",
      "level": <number or null>,
      "tolerance": <number — absolute price distance, never null for price_level conditions>,
      "indicator": "rsi" | "ema_fast" | "ema_slow" | "atr" | null,
      "indicator_params": {"period": <int>} | null,
      "timeframe": "<same as root timeframe, or override per-condition>",
      "notes": "<plain English: what this single condition represents>"
    }
  ],
  "logic": "ALL" | "ANY",
  "confirm_bars": <int — candle closes required to confirm trigger, minimum 1>,
  "window_minutes": <int — watch duration in minutes, or null if open-ended>,
  "on_timeout_event": "TIMEOUT" | null,
  "assumptions": ["<each default or inferred value you chose, one string per item>"]
}

════════════════════════════════════════
ALLOWED event_type VALUES — use EXACTLY one
════════════════════════════════════════
ENTRY_TRIGGER   — a level/retest/breakout just confirmed; ready to act
APPROACHING     — price is NEAR a level but not confirmed yet; heads-up only
INVALIDATED     — structure broke, level lost, thesis voided; stop watching
MANAGE_TRADE    — already in a position; something changed, user must act
TARGET_HIT      — take-profit / price objective reached
STOP_RISK       — price threatening the stop; adverse move; urgent
TIMEOUT         — watch window elapsed with no trigger
RANGE_STALL     — market is ranging / flat / doing nothing for the window
INDICATOR_STATE — an indicator condition met (RSI level, EMA cross, etc.)
INFO            — generic catch-all that fits none of the above

════════════════════════════════════════
OPERATOR REFERENCE — use EXACTLY one per condition
════════════════════════════════════════
touch_then_close_above — price enters level±tolerance THEN a candle CLOSES above level (retest support / bullish)
touch_then_close_below — price enters level±tolerance THEN a candle CLOSES below level (retest resistance / bearish)
close_above            — candle close strictly above level (breakout long; NOT a wick)
close_below            — candle close strictly below level (breakdown; NOT a wick)
crosses_above          — indicator value crosses above level from below
crosses_below          — indicator value crosses below level from above
gte                    — current value >= level (simple numeric threshold)
lte                    — current value <= level
stalls                 — price moves less than tolerance over window_minutes (range/flat)
expires                — time elapses: window_minutes reached with no other trigger firing

════════════════════════════════════════
RULES YOU MUST FOLLOW
════════════════════════════════════════

RULE 1 — CONFIRMATION LOGIC (most important):
  Never use a bare price touch as a trigger. Encode confirmation explicitly.
  • "retest [level] and hold"    → operator=touch_then_close_above (support) or touch_then_close_below (resistance) AND confirm_bars=2
  • "retest [level]"             → same operators, confirm_bars=1
  • "breakout / break above"     → operator=close_above, confirm_bars=1
  • "breakout and hold"          → operator=close_above, confirm_bars=2
  • "approaching / near / rises to / drops to"  → operator=gte (up) or lte (down), event_type=APPROACHING  [see Rule 10 for direction]
  If the goal is vague about confirmation, choose the trading-correct default above and record it in assumptions[].

RULE 2 — LEVELS:
  Only use numbers explicitly stated in the goal or provided in KEY_LEVELS context.
  NEVER invent or estimate a price level. If a required level is missing, set status="needs_clarification" and ask for it in clarification.

RULE 3 — TIMEFRAME:
  Default to 15m if not stated. Record the assumption. Use 1h for "hourly" goals, 5m for "5-minute" goals, 1m only if explicitly requested.

RULE 4 — TOLERANCE:
  Default to 0.05% of the level price (absolute). Example: level=160.50 → tolerance=round(160.50*0.0005, 3)=0.080.
  For JPY pairs the default applies without special multiplier (they are large numbers so 0.05% is already reasonable).
  Always record the tolerance assumption.

RULE 5 — INDICATORS:
  Supported: rsi (period default 14), ema_fast (period default 9), ema_slow (period default 21), atr (period default 14).
  If the goal references any other indicator, set status="needs_clarification".

RULE 6 — RANGE / FLAT:
  "ranging", "flat", "doing nothing", "stalls", "no movement" → operator=stalls, event_type=RANGE_STALL.
  tolerance for stalls = the acceptable total price range over the window (default: 0.1% of current price).

RULE 7 — TIME WINDOWS:
  Convert all time to minutes. "4 hours"=240, "2h"=120, "30 minutes"=30.
  "end of day" / "by close" → set status="needs_clarification" (timezone ambiguous).
  If no window stated, set window_minutes=null and on_timeout_event=null.

RULE 8 — COMPOUND GOALS:
  Multiple conditions → multiple objects in conditions[], logic=ALL by default.
  Use logic=ANY only if goal explicitly says "or / either".

RULE 9 — IN-POSITION GOALS:
  If user says "I'm long", "I'm short", "I'm in a position" → event_type=MANAGE_TRADE (or STOP_RISK if stop-specific).

RULE 10 — APPROACHING vs ENTRY_TRIGGER:
  Use APPROACHING (operator=lte or gte) when the goal is simply to be notified the moment price REACHES a level — no confirmation needed.
  Use ENTRY_TRIGGER when the goal requires a candle to CLOSE beyond the level or retest and hold.

  ⚠ DIRECTION IS CRITICAL — choosing the wrong operator silently misfires:
    lte fires when  price <= level  →  use ONLY when price must come DOWN to reach the level
    gte fires when  price >= level  →  use ONLY when price must go   UP  to reach the level

    DOWN-to-level patterns (use operator=lte):
      "drops to X / falls to X / dips to X / retreats to X / sells off to X"
      "down to X / decline to X / pull back to X / correct to X"

    UP-to-level patterns (use operator=gte):
      "rises to X / rallies to X / climbs to X / bounces to X / goes up to X"
      "up to X / advances to X / pushes to X / reaches X from below"

    Ambiguous direction ("reaches X / hits X / gets to X / at X / near X / approaching X"):
      → compare CURRENT_PRICE (from context) to X:
          CURRENT_PRICE < X  →  price must go UP  to reach X  →  operator=gte
          CURRENT_PRICE > X  →  price must come DOWN to reach X  →  operator=lte
          CURRENT_PRICE = X  →  already there; prefer gte for safety

  APPROACHING patterns (no confirmation language):
    "drops to X / falls to X / reaches X / hits X / gets to X / when price is at X"
    "rises to X / rallies to X / climbs to X"
    "alert me when X / ping me when X / notify when X" (bare level, no "and holds")
    "approaching X / near X / getting close to X / nearing X"

  ENTRY_TRIGGER patterns (explicit confirmation required):
    "retest X / retest and hold / bounce off X / breakout above X / close above X / close below X"
    "confirm / hold / wait for close / break and stay above" (any confirmation language)
    operator=touch_then_close_above/below or close_above/close_below.

  DEFAULT RULE: if no confirmation language is present, choose APPROACHING. Only use ENTRY_TRIGGER when the user explicitly asks for a candle close confirmation.

OUTPUT: Return ONLY the JSON object. No markdown. No prose. No code fences.
"""


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _build_user_message(goal_text: str, symbol: str,
                        current_price: float | None,
                        levels: dict | None) -> str:
    parts = [f"GOAL: {goal_text}", f"SYMBOL: {symbol}"]
    if current_price is not None:
        parts.append(f"CURRENT_PRICE: {current_price}")
    if levels:
        parts.append(f"KEY_LEVELS: {json.dumps(levels)}")
    return "\n".join(parts)


def _call_nim(messages: list[dict]) -> str:
    """POST to NIM chat/completions. Returns raw content string. Raises on error."""
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NIM_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},  # NIM JSON mode — enforces valid JSON output
    }
    resp = requests.post(
        f"{NIM_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _strip_and_parse(text: str) -> dict:
    """Strip any stray markdown fences then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return json.loads(text.strip())


def _validate(rule: dict) -> tuple[bool, str | None]:
    """Check required keys, allowed values, and numeric types."""
    missing = REQUIRED_RULE_KEYS - set(rule.keys())
    if missing:
        return False, f"Missing required keys: {missing}"
    if rule.get("status") not in ("ok", "needs_clarification"):
        return False, f"Invalid status: {rule.get('status')!r}"
    if rule.get("event_type") not in ALLOWED_EVENT_TYPES:
        return False, f"Invalid event_type: {rule.get('event_type')!r}"
    if rule.get("timeframe") not in ALLOWED_TIMEFRAMES:
        return False, f"Invalid timeframe: {rule.get('timeframe')!r}"
    if not isinstance(rule.get("conditions"), list):
        return False, "conditions must be a list"
    if rule.get("logic") not in ("ALL", "ANY"):
        return False, f"Invalid logic: {rule.get('logic')!r}"
    if not isinstance(rule.get("confirm_bars"), int):
        return False, "confirm_bars must be an integer"
    wm = rule.get("window_minutes")
    if wm is not None and not isinstance(wm, (int, float)):
        return False, "window_minutes must be a number or null"
    return True, None


def _fallback(symbol: str, reason: str) -> dict:
    return {
        "status": "needs_clarification",
        "clarification": reason,
        "symbol": symbol,
        "event_type": "INFO",
        "conditions": [],
        "assumptions": [],
    }


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def parse_goal(
    goal_text: str,
    symbol: str,
    current_price: float | None = None,
    levels: dict | None = None,
) -> dict:
    """
    Translate a natural-language watch goal into a structured rule dict.

    Args:
        goal_text:     The user's natural-language goal, e.g. "wait for USDJPY retest 160.50".
        symbol:        The ticker as typed by the user, e.g. "USDJPY". Passed to the model as context.
        current_price: Optional current market price (helps model calibrate tolerances).
        levels:        Optional dict of key price levels, e.g. {"support": 160.50, "stop": 159.80}.

    Returns:
        A dict with at minimum {"status", "clarification", "symbol", "event_type", "conditions"}.
        status="ok"                 → rule is ready; hand it to the watch loop.
        status="needs_clarification"→ ask the user the question in "clarification".
        Never raises — always returns a dict.
    """
    if not NIM_API_KEY:
        return _fallback(symbol, "NIM_API_KEY is not set.")

    user_content = _build_user_message(goal_text, symbol, current_price, levels)
    base_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    raw = None

    for attempt in range(1, 3):  # max 2 attempts
        # On retry, append the bad response and a correction instruction
        if attempt == 2 and raw is not None:
            call_messages = base_messages + [
                {"role": "assistant", "content": raw},
                {"role": "user",      "content":
                    "That response was not valid JSON matching the schema. "
                    "Return ONLY the JSON object. No markdown. No prose. No fences."},
            ]
        else:
            call_messages = base_messages

        # --- Transport ---
        try:
            raw = _call_nim(call_messages)
        except requests.exceptions.Timeout:
            print(f"[goal_parser] NIM timeout (attempt {attempt}/2)", file=sys.stderr)
            if attempt < 2:
                time.sleep(2)
                continue
            return _fallback(symbol, "NIM request timed out — try again.")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"[goal_parser] NIM HTTP {code} (attempt {attempt}/2): {e}", file=sys.stderr)
            if attempt < 2 and code in (429, 500, 502, 503):
                time.sleep(3)
                continue
            return _fallback(symbol, f"NIM API error ({code}) — check your NIM_API_KEY and model ID.")
        except requests.RequestException as e:
            print(f"[goal_parser] NIM network error (attempt {attempt}/2): {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2)
                continue
            return _fallback(symbol, "Network error reaching NIM — check connection.")

        # --- Parse JSON ---
        try:
            rule = _strip_and_parse(raw)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"[goal_parser] JSON parse error (attempt {attempt}/2): {e}", file=sys.stderr)
            if attempt < 2:
                continue
            return _fallback(symbol, "Parser could not produce valid JSON — please restate the goal.")

        # --- Validate schema ---
        valid, err = _validate(rule)
        if not valid:
            print(f"[goal_parser] Schema validation error (attempt {attempt}/2): {err}", file=sys.stderr)
            raw = json.dumps(rule)  # feed the bad output back on retry
            if attempt < 2:
                continue
            return _fallback(symbol, "Parser could not produce a valid rule — please restate the goal.")

        return rule

    # Should not reach here, but safety net
    return _fallback(symbol, "Parser failed after retries.")


# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    tests = [
        {
            "label": "1 — Retest with timeout",
            "goal":  "wait for USDJPY to retest 160.50 and hold, give up after 4 hours",
            "symbol": "USDJPY=X",
            "price":  161.20,
            "levels": None,
        },
        {
            "label": "2 — In-position stop management",
            "goal":  "I'm long GOLD, tell me if price drops near 4200 so I can manage the stop",
            "symbol": "GC=F",
            "price":  4250.0,
            "levels": {"support": 4200, "stop": 4180},
        },
        {
            "label": "3 — Range stall",
            "goal":  "ping me if EURUSD just ranges flat for an hour and does nothing",
            "symbol": "EURUSD=X",
            "price":  1.0850,
            "levels": None,
        },
    ]

    for t in tests:
        print(f"\n{'═'*64}")
        print(f"TEST {t['label']}")
        print(f"GOAL: {t['goal']}")
        print(f"{'─'*64}")
        result = parse_goal(
            t["goal"], t["symbol"],
            current_price=t["price"],
            levels=t["levels"],
        )
        pprint.pprint(result, width=80, sort_dicts=False)

    print(f"\n{'═'*64}")
    print("Done. Check that:")
    print("  • Test 1 → event_type=ENTRY_TRIGGER, operator=touch_then_close_above, confirm_bars>=2, window_minutes=240")
    print("  • Test 2 → event_type=MANAGE_TRADE (or STOP_RISK), operator=lte, level=4200")
    print("  • Test 3 → event_type=RANGE_STALL, operator=stalls, window_minutes=60")
