"""
notify.py — Phone notification module for the trading watchdog.
Transport: ntfy.sh (https://ntfy.sh). No account or API key required.
Single dependency: requests.

Usage:
    from notify import notify_event, EventType

    notify_event(EventType.ENTRY_TRIGGER, "BTCUSDT", "retest confirmed at 65,400", current_price=65412.5)
    notify_event(EventType.STOP_RISK, "ETHUSDT", "adverse move accelerating", current_price=3480.0, priority_override="urgent")
"""

import os
import sys
import time
import warnings
from enum import Enum

import requests

# ──────────────────────────────────────────────────────────────
# CONFIG — edit here or set env vars
# ──────────────────────────────────────────────────────────────
NTFY_SERVER = "https://ntfy.sh"

_PLACEHOLDER = "CHANGE_ME_to_a_long_random_secret_topic"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", _PLACEHOLDER)

if NTFY_TOPIC == _PLACEHOLDER:
    warnings.warn(
        "[notify] NTFY_TOPIC is still the placeholder. "
        "Set the NTFY_TOPIC environment variable to a long random string "
        "before relying on notifications.",
        stacklevel=2,
    )

# ──────────────────────────────────────────────────────────────
# EVENT TAXONOMY
# Each entry: (human_label, default_priority, ntfy_tag, emoji_prefix)
# ──────────────────────────────────────────────────────────────
class EventType(str, Enum):
    ENTRY_TRIGGER   = "ENTRY_TRIGGER"
    APPROACHING     = "APPROACHING"
    INVALIDATED     = "INVALIDATED"
    MANAGE_TRADE    = "MANAGE_TRADE"
    TARGET_HIT      = "TARGET_HIT"
    STOP_RISK       = "STOP_RISK"
    TIMEOUT         = "TIMEOUT"
    RANGE_STALL     = "RANGE_STALL"
    INDICATOR_STATE = "INDICATOR_STATE"
    INFO            = "INFO"

# fmt: off
_EVENT_META: dict[str, tuple[str, str, str, str]] = {
    # event_type          human_label          priority    tag                   prefix
    EventType.ENTRY_TRIGGER:   ("setup ready",       "high",     "dart",               "🎯"),
    EventType.APPROACHING:     ("approaching level", "default",  "eyes",               "👀"),
    EventType.INVALIDATED:     ("setup invalidated", "high",     "x",                  "❌"),
    EventType.MANAGE_TRADE:    ("manage trade",      "high",     "wrench",             "🔧"),
    EventType.TARGET_HIT:      ("target hit",        "high",     "white_check_mark",   "✅"),
    EventType.STOP_RISK:       ("stop at risk",      "urgent",   "rotating_light",     "🚨"),
    EventType.TIMEOUT:         ("watch expired",     "low",      "hourglass",          "⏳"),
    EventType.RANGE_STALL:     ("stalled",           "low",      "zzz",                "😴"),
    EventType.INDICATOR_STATE: ("indicator signal",  "default",  "bar_chart",          "📊"),
    EventType.INFO:            ("info",              "default",  "information_source", "ℹ️"),
}
# fmt: on

_VALID_PRIORITIES = {"min", "low", "default", "high", "max"}


def _normalise_priority(priority: str) -> str:
    p = priority.strip().lower()
    if p == "urgent":
        return "max"
    return p if p in _VALID_PRIORITIES else "default"


# ──────────────────────────────────────────────────────────────
# CORE SEND
# ──────────────────────────────────────────────────────────────
def notify(
    title: str,
    message: str,
    priority: str = "default",
    tags: list[str] | None = None,
    click_url: str | None = None,
) -> bool:
    """POST a notification to ntfy.sh. Returns True on success, never raises."""
    url = f"{NTFY_SERVER.rstrip('/')}/{NTFY_TOPIC}"
    headers = {
        "Title":    title,
        "Priority": _normalise_priority(priority),
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url

    redacted = f"{NTFY_SERVER.rstrip('/')}/<topic>"  # never log the real topic

    for attempt in range(1, 3):  # max 2 attempts
        try:
            resp = requests.post(url, data=message.encode(), headers=headers, timeout=10)
            if resp.ok:
                return True
            print(
                f"[notify] HTTP {resp.status_code} from {redacted} "
                f"(attempt {attempt}/2)",
                file=sys.stderr,
            )
        except requests.RequestException as exc:
            print(
                f"[notify] Network error (attempt {attempt}/2): {exc}",
                file=sys.stderr,
            )
        if attempt < 2:
            time.sleep(1)

    return False


# ──────────────────────────────────────────────────────────────
# EVENT DISPATCHER
# ──────────────────────────────────────────────────────────────
def notify_event(
    event_type: EventType | str,
    symbol: str,
    message: str,
    current_price: float | None = None,
    click_url: str | None = None,
    priority_override: str | None = None,
) -> bool:
    """
    Dispatch a typed trading event notification.

    Args:
        event_type:        One of EventType members (or its string value).
        symbol:            Instrument ticker, e.g. "BTCUSDT".
        message:           Human-readable description of what triggered the alert.
        current_price:     If provided, appended to the message body.
        click_url:         Optional URL opened when the notification is tapped.
        priority_override: Overrides the event type's default priority.
    """
    try:
        key = EventType(event_type) if not isinstance(event_type, EventType) else event_type
    except ValueError:
        key = EventType.INFO

    human_label, default_priority, tag, prefix = _EVENT_META[key]
    priority = _normalise_priority(priority_override) if priority_override else default_priority

    # Title must be Latin-1 safe (HTTP header); emoji goes in the body instead.
    title = f"{symbol} - {human_label}"

    body = f"{prefix} {message}"
    if current_price is not None:
        body = f"{prefix} {message}\nPrice: {current_price:,.2f}"

    return notify(title, body, priority=priority, tags=[tag], click_url=click_url)


# ──────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sending test notifications …")

    tests = [
        (EventType.ENTRY_TRIGGER,   "BTCUSDT",  "retest of 65,400 confirmed - entry zone active", 65_412.50),
        (EventType.STOP_RISK,       "ETHUSDT",  "price dropped below support — check stop",        3_480.00),
        (EventType.TARGET_HIT,      "SOLUSDT",  "TP1 reached at 175",                              175.10),
        (EventType.RANGE_STALL,     "BNBUSDT",  "no movement for 2 h — watching",                  None),
        (EventType.INFO,            "WATCHDOG", "heartbeat — notifier is working",                 None),
    ]

    for et, sym, msg, price in tests:
        ok = notify_event(et, sym, msg, current_price=price)
        status = "✓ sent" if ok else "✗ failed"
        print(f"  {status}  [{et.value}] {sym}")
        time.sleep(1)

    print("Done. Check your phone.")
