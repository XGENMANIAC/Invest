"""
watchdog.py — Interactive entry point for the trading watchdog app.

BOUNDARY: This runner orchestrates only. It never places or modifies trades.
When a watch triggers, confirm the setup manually before executing.

Usage:
    python watchdog.py

Commands (type at the prompt):
    watch <SYMBOL> <goal text...> [--keep]
    list | show <id> | cancel <id> | cancel all | quit | help
"""

import json
import sys
import threading
import time
import warnings
from datetime import datetime

# Suppress at-import warnings from sub-modules — we handle them in _check_env()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from goal_parser import parse_goal, NIM_API_KEY, NIM_BASE_URL, NIM_MODEL
    from notify import notify_event, NTFY_TOPIC, _PLACEHOLDER
    from watch_loop import (
        SYMBOL_TO_TICKER,
        TIMEFRAME_TO_YF,
        _check_approaching,
        _fetch_candles,
        _is_fresh,
        _resolve_ticker,
        evaluate_conditions,
    )

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_POLL_SECONDS = 20   # seconds between ticks per watch
DEFAULT_HEADS_UP     = True  # fire APPROACHING events when price nears a level

# ── SHARED STATE ───────────────────────────────────────────────────────────────
_registry: dict[str, dict] = {}  # wid → watch entry
_registry_lock = threading.Lock()
_watch_counter = 0  # protected by _registry_lock

# ── UTILITIES ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _prompt() -> None:
    print("\nwatchdog> ", end="", flush=True)


def _get_current_price(symbol: str) -> float | None:
    """Fetch latest price for symbol using the watch_loop data layer. Returns None on failure."""
    try:
        ticker   = _resolve_ticker(symbol)
        interval, yf_range = TIMEFRAME_TO_YF.get("1m", ("1m", "1d"))
        candles  = _fetch_candles(ticker, interval, yf_range)
        if candles and candles.close:
            return float(candles.close[-1])
    except Exception:
        pass
    return None


def _remove_watch(wid: str) -> None:
    with _registry_lock:
        _registry.pop(wid, None)

# ── BACKGROUND WATCH THREAD ────────────────────────────────────────────────────
# Stoppable version of watch_loop.run_watch — uses stop_event.wait() instead of
# time.sleep() so cancel/quit can interrupt a sleeping tick within ~1 second.

def _watch_thread(entry: dict) -> None:
    wid          = entry["id"]
    rule         = entry["rule"]
    stop_event   = entry["stop_event"]
    poll_seconds = entry["poll_seconds"]
    heads_up     = entry["heads_up"]
    one_shot     = entry["one_shot"]
    symbol       = rule["symbol"]
    timeframe    = rule.get("timeframe", "15m")
    window_m     = rule.get("window_minutes")
    timeout_ev   = rule.get("on_timeout_event", "TIMEOUT")

    try:
        ticker = _resolve_ticker(symbol)
    except ValueError as exc:
        _log(f"[{wid}] Cannot start watch: {exc}")
        _remove_watch(wid)
        _prompt()
        return

    interval, yf_range = TIMEFRAME_TO_YF.get(timeframe, ("15m", "5d"))
    start    = time.time()
    deadline = start + window_m * 60 if window_m else None
    heads_up_fired: set = set()
    tick = 0

    _log(f"[{wid}] Started: {symbol} | {timeframe} | {rule.get('human_summary', '')}")
    _prompt()

    while not stop_event.is_set():
        tick += 1
        now         = time.time()
        elapsed_min = (now - start) / 60

        # ── Timeout ───────────────────────────────────────────────────────
        if deadline and now >= deadline:
            _log(f"[{wid}] Tick {tick} | Deadline reached ({elapsed_min:.1f}min) → {timeout_ev}")
            if timeout_ev:
                try:
                    notify_event(
                        timeout_ev, symbol,
                        f"[TIMEOUT — NOT A TRIGGER] Watch {wid} expired after "
                        f"{elapsed_min:.0f}min. Condition was never met. No action needed.",
                    )
                except Exception as exc:
                    _log(f"[{wid}] Notify error: {exc}")
            break

        # ── Fetch ─────────────────────────────────────────────────────────
        try:
            candles = _fetch_candles(ticker, interval, yf_range)
        except Exception as exc:
            _log(f"[{wid}] Tick {tick} | Fetch error: {exc} — skipping")
            candles = None

        if candles is None:
            _log(f"[{wid}] Tick {tick} | No data — retrying next tick")
            _prompt()
            stop_event.wait(poll_seconds)
            continue

        # ── Staleness guard — skip evaluation on frozen/weekend data ──────
        fresh, age_min = _is_fresh(candles, timeframe)
        if not fresh:
            _log(
                f"[{wid}] Tick {tick} | Data stale ({age_min:.0f}min old)"
                f" — market likely closed. Skipping evaluation, will retry."
            )
            _prompt()
            stop_event.wait(poll_seconds)
            continue

        # ── Evaluate ──────────────────────────────────────────────────────
        try:
            result = evaluate_conditions(rule, candles)
        except Exception as exc:
            _log(f"[{wid}] Tick {tick} | Eval error: {exc} — skipping")
            _prompt()
            stop_event.wait(poll_seconds)
            continue

        cur    = result["current_price"]
        trig   = result["triggered"]
        detail = result["detail"]
        tl     = f"{(deadline - now) / 60:.0f}min left" if deadline else "open-ended"

        _log(
            f"[{wid}] Tick {tick} | {symbol} @ {cur:.5f} | {elapsed_min:.1f}min"
            f" ({tl}) | {'*** TRIGGERED ***' if trig else 'watching'} | {detail}"
        )
        _prompt()

        # ── Approaching heads-up ──────────────────────────────────────────
        if heads_up and not trig:
            try:
                _check_approaching(rule, candles, heads_up_fired)
            except Exception as exc:
                _log(f"[{wid}] Approaching check error: {exc}")

        # ── Trigger ───────────────────────────────────────────────────────
        if trig:
            event_type = rule.get("event_type", "INFO")
            candle_ts  = result.get("candle_time")
            if candle_ts:
                from datetime import timezone
                ct = datetime.fromtimestamp(candle_ts, tz=timezone.utc).strftime("%H:%M UTC")
                price_note = (
                    f"Triggered on {timeframe} candle closed at {ct}: {cur:.5f}. "
                    f"Verify current live price before acting."
                )
            else:
                price_note = f"Trigger price (candle close): {cur:.5f}. Verify current live price before acting."
            notify_body = f"{rule.get('human_summary', detail)}\n{price_note}"
            try:
                notify_event(event_type, symbol, notify_body, current_price=cur)
            except Exception as exc:
                _log(f"[{wid}] Notify error: {exc}")
            _log(f"[{wid}] Alert sent — {event_type} for {symbol} @ {cur:.5f}")
            _prompt()
            if one_shot:
                _log(f"[{wid}] one_shot=True — watch complete.")
                _prompt()
                break

        # ── Interruptible sleep ───────────────────────────────────────────
        sleep_secs = poll_seconds
        if deadline:
            remaining  = deadline - time.time()
            sleep_secs = min(poll_seconds, max(1.0, remaining))
        stop_event.wait(sleep_secs)

    if stop_event.is_set():
        _log(f"[{wid}] Watch cancelled.")
        _prompt()

    _remove_watch(wid)

# ── COMMAND HANDLERS ───────────────────────────────────────────────────────────

def _cmd_watch(symbol: str, goal_text_with_flags: str) -> None:
    symbol = symbol.upper()

    # Parse --keep flag
    one_shot = True
    if "--keep" in goal_text_with_flags:
        one_shot  = False
        goal_text = goal_text_with_flags.replace("--keep", "").strip()
    else:
        goal_text = goal_text_with_flags.strip()

    if not goal_text:
        print("  Usage: watch <SYMBOL> <goal text...> [--keep]")
        return

    # Validate symbol early
    try:
        _resolve_ticker(symbol)
    except ValueError as exc:
        print(f"  {exc}")
        return

    # Current price (best-effort)
    print(f"  Fetching {symbol} price...", end="", flush=True)
    current_price = _get_current_price(symbol)
    print(f" {current_price:.5f}" if current_price else " (unavailable)")

    # Parse goal via LLM (this is the one call to goal_parser)
    print("  Parsing goal via Kimi K2.6...", end="", flush=True)
    try:
        rule = parse_goal(goal_text, symbol, current_price=current_price)
    except Exception as exc:
        print(f"\n  Parser error: {exc}")
        return
    print(" done.")

    if rule.get("status") == "needs_clarification":
        print(f"\n  Needs clarification: {rule.get('clarification')}")
        print("  Rephrase your goal with that detail and try again.\n")
        return

    # Assign ID and build entry
    with _registry_lock:
        global _watch_counter
        _watch_counter += 1
        wid = f"w{_watch_counter}"

    stop_event = threading.Event()
    entry = {
        "id":           wid,
        "symbol":       symbol,
        "rule":         rule,
        "stop_event":   stop_event,
        "start_time":   time.time(),
        "one_shot":     one_shot,
        "poll_seconds": DEFAULT_POLL_SECONDS,
        "heads_up":     DEFAULT_HEADS_UP,
    }
    thread = threading.Thread(
        target=_watch_thread,
        args=(entry,),
        daemon=True,
        name=f"watch-{wid}",
    )
    entry["thread"] = thread

    with _registry_lock:
        _registry[wid] = entry

    thread.start()

    # Confirmation
    print()
    print(f"  [{wid}] {rule.get('human_summary', goal_text)}")
    print(f"        event_type : {rule.get('event_type')}")
    print(f"        timeframe  : {rule.get('timeframe')}")
    wm = rule.get("window_minutes")
    print(f"        window     : {wm}min" if wm else "        window     : open-ended")
    print(f"        one_shot   : {one_shot} (add --keep to keep watching after trigger)")
    for assumption in rule.get("assumptions", []):
        print(f"        assumed    : {assumption}")
    print(f"\n  Watch {wid} is running in the background.\n")


def _cmd_list() -> None:
    with _registry_lock:
        entries = list(_registry.values())

    if not entries:
        print("  No active watches.")
        return

    now = time.time()
    print()
    print(f"  {'ID':<5}  {'SYMBOL':<9}  {'EVENT_TYPE':<18}  {'TIME LEFT':<11}  SUMMARY")
    print(f"  {'-'*5}  {'-'*9}  {'-'*18}  {'-'*11}  {'-'*35}")
    for e in entries:
        rule = e["rule"]
        wm   = rule.get("window_minutes")
        if wm:
            rem = (e["start_time"] + wm * 60) - now
            tl  = f"{max(0, rem / 60):.0f}min" if rem > 0 else "expired"
        else:
            tl  = "open-ended"
        summary = (rule.get("human_summary") or "")[:38]
        print(f"  {e['id']:<5}  {e['symbol']:<9}  {rule.get('event_type',''):<18}  {tl:<11}  {summary}")
    print()


def _cmd_show(wid: str) -> None:
    with _registry_lock:
        entry = _registry.get(wid)

    if entry is None:
        print(f"  No active watch '{wid}'. Use 'list' to see what's running.")
        return

    rule = entry["rule"]
    now  = time.time()
    wm   = rule.get("window_minutes")
    tl   = (
        f"{max(0, (entry['start_time'] + wm * 60 - now) / 60):.0f}min"
        if wm else "open-ended"
    )

    print()
    print(f"  Watch {wid}  |  {entry['symbol']}  |  time left: {tl}")
    print(json.dumps(rule, indent=4, default=str))
    print()


def _cmd_cancel(wid: str) -> None:
    with _registry_lock:
        entry = _registry.get(wid)

    if entry is None:
        print(f"  No active watch '{wid}'. Use 'list' to see what's running.")
        return

    entry["stop_event"].set()
    print(f"  Cancelling {wid}...")
    entry["thread"].join(timeout=5)
    print(f"  {wid} stopped.")


def _cmd_cancel_all() -> None:
    with _registry_lock:
        entries = list(_registry.values())

    if not entries:
        print("  No active watches.")
        return

    for e in entries:
        e["stop_event"].set()

    print(f"  Cancelling {len(entries)} watch(es)...")
    for e in entries:
        e["thread"].join(timeout=5)
    print("  All watches stopped.")


def _cmd_help() -> None:
    print("""
  Commands
  ─────────────────────────────────────────────────────────────
  watch <SYMBOL> <goal text...>    parse goal, start a background watch
        [--keep]                   keep watching after trigger (default: stop)

  list                             show all active watches + time left
  show  <id>                       print the full parsed rule (verify the AI understood)
  cancel <id>                      stop one watch  (e.g. cancel w1)
  cancel all                       stop all watches
  quit                             stop all watches and exit
  help                             show this message

  Examples
  ─────────────────────────────────────────────────────────────
  watch USDJPY wait for retest of 160.50 and hold, give up after 4h
  watch GOLD   tell me if price drops near 2400 so I can manage the stop --keep
  watch EURUSD ping me if it just ranges flat for an hour
  show w1
  cancel w2
""")

# ── STARTUP ─────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    topic_ok = bool(NTFY_TOPIC and NTFY_TOPIC != _PLACEHOLDER)
    key_ok   = bool(NIM_API_KEY)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Trading Watchdog                            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Notifier  : ntfy.sh topic {'[SET]    ' if topic_ok else '[NOT SET]'}                    ║")
    print(f"║  Parser    : {NIM_MODEL:<20} via NIM  ║")
    print(f"║  API key   : {'[SET]' if key_ok else '[NOT SET — parsing disabled]'}                            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  This app watches and notifies only. It never places or ║")
    print("║  modifies trades. When a watch triggers, confirm the    ║")
    print("║  setup before executing manually.                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _check_env() -> None:
    warnings_found = []
    if not NIM_API_KEY:
        warnings_found.append("NIM_API_KEY is not set  →  export NIM_API_KEY=your_nvidia_nim_key")
    if not NTFY_TOPIC or NTFY_TOPIC == _PLACEHOLDER:
        warnings_found.append("NTFY_TOPIC is not set   →  export NTFY_TOPIC=your_secret_topic")

    if warnings_found:
        print("  [!] Missing environment variables:")
        for w in warnings_found:
            print(f"      {w}")
        print()

# ── MAIN LOOP ──────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner()
    _check_env()
    print("  Type 'help' for commands.\n")

    while True:
        try:
            print("watchdog> ", end="", flush=True)
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _cmd_cancel_all()
            print("  Goodbye.")
            break

        if not line:
            continue

        parts = line.split()
        cmd   = parts[0].lower()

        try:
            if cmd == "watch":
                if len(parts) < 3:
                    print("  Usage: watch <SYMBOL> <goal text...> [--keep]")
                else:
                    _cmd_watch(parts[1], " ".join(parts[2:]))

            elif cmd == "list":
                _cmd_list()

            elif cmd == "show":
                if len(parts) != 2:
                    print("  Usage: show <id>  (e.g. show w1)")
                else:
                    _cmd_show(parts[1])

            elif cmd == "cancel":
                if len(parts) != 2:
                    print("  Usage: cancel <id> | cancel all")
                elif parts[1].lower() == "all":
                    _cmd_cancel_all()
                else:
                    _cmd_cancel(parts[1])

            elif cmd in ("quit", "exit", "q"):
                _cmd_cancel_all()
                print("  Goodbye.")
                break

            elif cmd == "help":
                _cmd_help()

            else:
                print(f"  Unknown command '{cmd}'. Type 'help'.")

        except Exception as exc:
            print(f"  Error: {exc}")


if __name__ == "__main__":
    main()
