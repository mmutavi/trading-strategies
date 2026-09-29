"""
Pairs Trading (Stat-Arb Mean Reversion) - LIVE (paper) bot for Alpaca.

Places REAL orders against your Alpaca account (paper by default).
Meant to run ALONGSIDE orb_live.py / vwap_live.py / vp_live.py - different
logic entirely, its own log file, its own two-symbol position.

Strategy:
- Track the price RATIO of two correlated symbols (default SPY / QQQ).
- Every RECALC_SECONDS, recompute the ratio's rolling mean and standard
  deviation over the last LOOKBACK_MINUTES of 1-minute bars.
- z-score = (current ratio - rolling mean) / rolling stdev.
- If z >= ENTRY_Z: the ratio is abnormally HIGH -> A is rich vs B ->
  SHORT A, LONG B, betting the ratio falls back toward its mean.
- If z <= -ENTRY_Z: the ratio is abnormally LOW -> A is cheap vs B ->
  LONG A, SHORT B.
- Take profit when the ratio reverts to within EXIT_Z of its mean.
- Hard stop when the ratio keeps diverging past HARD_STOP_Z (the two
  symbols have decoupled - the mean-reversion bet was wrong).
- Only one pair-trade open at a time. After a stop-out, the pair goes on
  cooldown before re-entering.
- Both legs are opened/closed together. If one leg's order fails after
  the other already filled, the bot immediately tries to unwind the leg
  that DID fill, so you're never left silently holding a naked, unhedged
  position - see enter_pair() / close_pair() for exactly how.
- Exit at 3:55pm ET if still open.
- This only works with two STOCK symbols (Alpaca doesn't support short-
  selling crypto, and one leg of every pair trade here is always a
  short).

Requires:
    pip install alpaca-py pytz colorama pandas

    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python pairs_live.py

Every trade (open and close) is appended to pairs_trade_log.jsonl in the
same folder - one JSON object per line.

Stop the bot any time with Ctrl+C - it will try to close any open
positions before exiting.
"""

import os
import json
import time
from datetime import datetime, time as dtime, timedelta
import pytz

from colorama import init as colorama_init, Fore, Style

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

colorama_init(autoreset=True)

# ---------------- CONFIG ----------------
# IMPORTANT: these must NOT overlap with symbols traded by your other bots
# (orb_live.py: SPY, DIA / vwap_live.py: QQQ, IWM / vp_live.py: BTC/USD).
# Alpaca nets all orders to one symbol into a single account-wide position,
# so if two bots trade the same symbol, neither one's "position" is really
# its own - closing what looks like "the position" can actually close (or
# partially close) another bot's trade out from under it. VOO and IVV are
# both S&P 500 index funds - about as close to true twins as a pairs trade
# gets, and untouched by the other three bots.
SYMBOL_A = "VOO"
SYMBOL_B = "IVV"
NOTIONAL_PER_LEG = 1000.0    # each leg sized to roughly this many dollars at entry

LOOKBACK_MINUTES = 60        # rolling window (in 1-min bars) for the ratio's mean/stdev
MIN_BARS_FOR_ZSCORE = 30     # need at least this many aligned bars before trusting the stats
RECALC_SECONDS = 60          # how often to refetch bars and recompute mean/stdev

ENTRY_Z = 2.0                # enter when |z-score| crosses this
EXIT_Z = 0.25                # take profit once the ratio reverts to within this of its mean
HARD_STOP_Z = 4.0            # stop out if the ratio keeps diverging past this

MAX_DAILY_LOSS_USD = 100.0
REENTRY_COOLDOWN_MINUTES = 10
POLL_INTERVAL_SECONDS = 15
PAPER = True                 # keep True until you deliberately decide otherwise
LOG_FILE = "pairs_trade_log.jsonl"

ET = pytz.timezone("US/Eastern")
MARKET_OPEN = dtime(9, 30)
EXIT_TIME = dtime(15, 55)
# -----------------------------------------

API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables first.")

trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# ---------------- LOGGING / DISPLAY ----------------

def log(msg):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_trade_event(record):
    record["timestamp"] = datetime.now(ET).isoformat()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log(f"WARNING: could not write to {LOG_FILE}: {e}")


def _box(lines, color, double=True):
    """Draw a clean unicode box around the given lines (no emoji, no
    rainbow - just crisp single/double-line borders)."""
    tl, tr, bl, br, h, v = ("╔", "╗", "╚", "╝", "═", "║") if double else ("┌", "┐", "└", "┘", "─", "│")
    width = max(len(line) for line in lines) + 2
    print(color + tl + h * width + tr)
    for line in lines:
        print(color + v + " " + line.ljust(width - 2) + " " + v)
    print(color + bl + h * width + br + Style.RESET_ALL)


def print_startup_banner():
    title = "P A I R S   T R A D E R"
    subtitle = f"{SYMBOL_A} / {SYMBOL_B}  ·  stat-arb mean reversion"
    _box([title, subtitle], Fore.CYAN, double=True)


def print_pair_open(direction, price_a, price_b, qty_a, qty_b, z):
    if direction == "short_a_long_b":
        line_a = f"  SHORT {SYMBOL_A:<6} x{qty_a:<5} @ ~{price_a:.2f}   ↓"
        line_b = f"  LONG  {SYMBOL_B:<6} x{qty_b:<5} @ ~{price_b:.2f}   ↑"
    else:
        line_a = f"  LONG  {SYMBOL_A:<6} x{qty_a:<5} @ ~{price_a:.2f}   ↑"
        line_b = f"  SHORT {SYMBOL_B:<6} x{qty_b:<5} @ ~{price_b:.2f}   ↓"
    _box(
        ["OPENED PAIR TRADE", line_a, line_b, f"  z-score at entry: {z:+.2f}"],
        Fore.CYAN,
        double=True,
    )


def print_pair_close(direction, entry_a, exit_a, entry_b, exit_b, pnl, z, reason):
    color = Fore.GREEN if pnl >= 0 else Fore.RED
    sign = "+" if pnl >= 0 else ""
    _box(
        [
            f"CLOSED PAIR TRADE ({reason})",
            f"  {SYMBOL_A}: {entry_a:.2f} -> {exit_a:.2f}",
            f"  {SYMBOL_B}: {entry_b:.2f} -> {exit_b:.2f}",
            f"  z-score at exit: {z:+.2f}",
            f"  P&L: {sign}${pnl:.2f}",
        ],
        color,
        double=True,
    )


def print_day_summary(day, total_pnl, trades_taken):
    color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    sign = "+" if total_pnl >= 0 else ""
    _box(
        [
            f"END OF DAY SUMMARY (PAIRS) - {day}",
            f"  Trades taken: {trades_taken}",
            f"  Total P&L: {sign}${total_pnl:.2f}",
        ],
        color,
        double=True,
    )


# ---------------- DATA / ORDER HELPERS ----------------

def get_intraday_bars(symbol, day, now):
    start = ET.localize(datetime.combine(day, MARKET_OPEN))
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=now, feed=DataFeed.IEX)
    return stock_data_client.get_stock_bars(req).df


def compute_ratio_stats(day, now):
    """Returns (mean, stdev, n_aligned_bars) using aligned 1-min closes for
    SYMBOL_A and SYMBOL_B over the last LOOKBACK_MINUTES. Returns
    (None, None, n) if there isn't enough aligned data yet."""
    try:
        bars_a = get_intraday_bars(SYMBOL_A, day, now)
        bars_b = get_intraday_bars(SYMBOL_B, day, now)
    except Exception as e:
        log(f"WARNING: could not fetch bars for ratio stats: {e}")
        return None, None, 0

    if bars_a is None or bars_a.empty or bars_b is None or bars_b.empty:
        return None, None, 0

    bars_a = bars_a.reset_index()
    bars_b = bars_b.reset_index()
    ts_col_a = "timestamp" if "timestamp" in bars_a.columns else bars_a.columns[1]
    ts_col_b = "timestamp" if "timestamp" in bars_b.columns else bars_b.columns[1]

    merged = bars_a[[ts_col_a, "close"]].merge(
        bars_b[[ts_col_b, "close"]], left_on=ts_col_a, right_on=ts_col_b, suffixes=("_a", "_b")
    )
    n = len(merged)
    if n < MIN_BARS_FOR_ZSCORE:
        return None, None, n

    merged = merged.tail(LOOKBACK_MINUTES)
    if (merged["close_b"] == 0).any():
        return None, None, n

    ratios = merged["close_a"] / merged["close_b"]
    mean = float(ratios.mean())
    stdev = float(ratios.std(ddof=0))
    if stdev <= 0:
        return None, None, n
    return mean, stdev, n


def get_latest_price(symbol):
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quote = stock_data_client.get_stock_latest_quote(req)[symbol]
    except Exception as e:
        log(f"WARNING: could not fetch quote for {symbol}: {e}")
        return None

    if quote.bid_price is None or quote.ask_price is None:
        return None
    if quote.bid_price <= 0 or quote.ask_price <= 0:
        return None
    return (quote.bid_price + quote.ask_price) / 2


def place_leg_order(symbol, side, qty):
    order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    return trading_client.submit_order(order)


def get_open_position_qty(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception as e:
        msg = str(e).lower()
        if "position does not exist" not in msg and "404" not in msg and "not found" not in msg:
            log(f"WARNING: unexpected error checking position for {symbol} ({e}). Treating as no position.")
        return 0.0


def close_leg(symbol, retries=0, retry_delay=1.0):
    """Closes whatever position currently exists in `symbol`, if any.
    Returns the signed qty that was closed, None if there was nothing to
    close, or False if a close order was attempted and failed.

    retries/retry_delay: used right after a fresh entry, where an order
    just submitted may not have registered as a position yet (fill isn't
    always instant). Without this, checking immediately could see "no
    position" and wrongly conclude there's nothing to unwind, while an
    order is still quietly working its way to a fill. Normal closes (of
    positions that have existed for a while) don't need this - retries=0
    there is fine and keeps the polling loop snappy."""
    qty = get_open_position_qty(symbol)
    attempt = 0
    while qty == 0 and attempt < retries:
        time.sleep(retry_delay)
        qty = get_open_position_qty(symbol)
        attempt += 1

    if qty == 0:
        return None
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    try:
        order = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY)
        trading_client.submit_order(order)
        log(f"Closed position of {qty} in {symbol}.")
        return qty
    except Exception as e:
        log(f"CRITICAL: failed to close position in {symbol}: {e}. CHECK THIS POSITION MANUALLY ON ALPACA.")
        return False


# ---------------- PAIR ENTRY / EXIT (with partial-fill protection) ----------------

def enter_pair(direction, price_a, price_b):
    """Opens both legs. If leg A fills but leg B fails, immediately
    unwinds leg A so we never end up silently holding one naked leg.
    Returns (qty_a, qty_b) on success, or (None, None) if no position
    was opened (or only a failed unwind attempt remains - logged as
    CRITICAL either way, since that needs a human to check)."""
    qty_a = max(1, round(NOTIONAL_PER_LEG / price_a))
    qty_b = max(1, round(NOTIONAL_PER_LEG / price_b))

    side_a = OrderSide.SELL if direction == "short_a_long_b" else OrderSide.BUY
    side_b = OrderSide.BUY if direction == "short_a_long_b" else OrderSide.SELL

    try:
        place_leg_order(SYMBOL_A, side_a, qty_a)
    except Exception as e:
        log(f"WARNING: leg A entry failed ({SYMBOL_A}): {e}. Aborting - no position opened.")
        return None, None

    try:
        place_leg_order(SYMBOL_B, side_b, qty_b)
    except Exception as e:
        log(f"WARNING: leg B entry failed ({SYMBOL_B}): {e}. Unwinding leg A to avoid a naked position.")
        # retries=3: leg A's order may not have registered as a filled
        # position instantly - give it a few seconds before concluding
        # there's nothing to unwind.
        unwind_result = close_leg(SYMBOL_A, retries=3, retry_delay=1.0)
        if unwind_result is False:
            log(f"CRITICAL: leg A ({SYMBOL_A}) could NOT be unwound after leg B failed. "
                f"You are holding a naked {SYMBOL_A} position. Check Alpaca immediately.")
        elif unwind_result is None:
            log(f"WARNING: leg A ({SYMBOL_A}) showed no open position to unwind - "
                f"its order may not have filled at all. Verify on Alpaca.")
        return None, None

    return qty_a, qty_b


def close_pair():
    """Closes both legs. Returns (qty_a_closed, qty_b_closed) - either
    value is None if that leg had nothing open, or False if closing that
    leg failed (already logged as CRITICAL by close_leg)."""
    closed_a = close_leg(SYMBOL_A)
    closed_b = close_leg(SYMBOL_B)
    return closed_a, closed_b


def close_pair_with_retries(max_attempts=5, delay_seconds=3):
    """Used only for the end-of-day forced flatten, where there are no
    more polling cycles left to naturally retry on. Keeps attempting
    both legs until both succeed (or return None, meaning already
    closed), up to max_attempts, then gives up and logs CRITICAL."""
    closed_a = closed_b = None
    for attempt in range(max_attempts):
        closed_a, closed_b = close_pair()
        if closed_a is not False and closed_b is not False:
            return closed_a, closed_b
        log(f"Retrying end-of-day pair close (attempt {attempt + 1}/{max_attempts})...")
        time.sleep(delay_seconds)
    log("CRITICAL: could not fully close the pair after repeated end-of-day attempts. "
        "MANUAL INTERVENTION NEEDED - check both legs on Alpaca.")
    return closed_a, closed_b


def leg_pnl(direction_of_leg, entry_price, exit_price, qty):
    if direction_of_leg == "long":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


# ---------------- CORE TRADING LOGIC ----------------

def run_trading_day(day):
    log(f"--- Starting Pairs trading day {day} ---")

    state = {
        "direction": None,
        "entry_price_a": None,
        "entry_price_b": None,
        "qty_a": None,
        "qty_b": None,
        "entry_z": None,
        "cooldown_until": None,
        "closing": False,        # True once we've decided to exit but haven't confirmed both legs closed yet
        "close_reason": None,
        "close_z": None,
    }
    ratio_mean = None
    ratio_stdev = None
    next_recalc = None
    stats_ready_logged = False

    exit_dt = ET.localize(datetime.combine(day, EXIT_TIME))
    trades_taken = 0
    total_pnl = 0.0

    while datetime.now(ET) < exit_dt:
        now = datetime.now(ET)
        loss_limit_hit = total_pnl <= -MAX_DAILY_LOSS_USD

        # Skip refetching ratio stats while we're just retrying a stuck
        # close - that decision is already locked in (state["close_z"]),
        # and we don't want an unrelated data hiccup on the stats fetch
        # to slow down unwinding a position that needs to close NOW.
        if not state["closing"] and (next_recalc is None or now >= next_recalc):
            ratio_mean, ratio_stdev, n_bars = compute_ratio_stats(day, now)
            if ratio_mean is not None and not stats_ready_logged:
                log(f"Ratio stats ready ({SYMBOL_A}/{SYMBOL_B}): mean={ratio_mean:.4f}, "
                    f"stdev={ratio_stdev:.5f} (from {n_bars} aligned bars)")
                stats_ready_logged = True
            next_recalc = now + timedelta(seconds=RECALC_SECONDS)

        price_a = get_latest_price(SYMBOL_A)
        price_b = get_latest_price(SYMBOL_B)

        if price_a is None or price_b is None or price_b == 0:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if state["direction"] is not None and state["closing"]:
            z = None  # exit decision already locked in - don't need a fresh z-score to retry it
        elif ratio_mean is None or ratio_stdev is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        else:
            current_ratio = price_a / price_b
            z = (current_ratio - ratio_mean) / ratio_stdev

        try:
            if state["direction"] is None:
                if loss_limit_hit:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                if state["cooldown_until"] is not None:
                    if now < state["cooldown_until"]:
                        time.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    state["cooldown_until"] = None

                direction = None
                if z >= ENTRY_Z:
                    direction = "short_a_long_b"
                elif z <= -ENTRY_Z:
                    direction = "long_a_short_b"

                if direction is not None:
                    qty_a, qty_b = enter_pair(direction, price_a, price_b)
                    if qty_a is not None and qty_b is not None:
                        state["direction"] = direction
                        state["entry_price_a"] = price_a
                        state["entry_price_b"] = price_b
                        state["qty_a"] = qty_a
                        state["qty_b"] = qty_b
                        state["entry_z"] = z
                        print_pair_open(direction, price_a, price_b, qty_a, qty_b, z)
                        log_trade_event({
                            "event": "open", "direction": direction,
                            "price_a": price_a, "price_b": price_b,
                            "qty_a": qty_a, "qty_b": qty_b, "z": z,
                        })

            else:
                direction = state["direction"]

                if not state["closing"]:
                    hit_target = abs(z) <= EXIT_Z
                    hit_stop = abs(z) >= HARD_STOP_Z
                    if hit_target or hit_stop:
                        state["closing"] = True
                        state["close_reason"] = "stopped out" if hit_stop else "target hit (reverted)"
                        state["close_z"] = z

                if state["closing"]:
                    closed_a, closed_b = close_pair()

                    if closed_a is False or closed_b is False:
                        # One or both legs failed to close. Do NOT mark the pair
                        # as flat and do NOT log a P&L number yet - that would
                        # misrepresent a leg that's still actually open. Stay in
                        # "closing" state and the next poll cycle will try again;
                        # any leg that DID succeed here is already flat, so a
                        # retry only re-attempts the leg that's still stuck.
                        log("CRITICAL: pair close incomplete this cycle - will keep retrying. "
                            "NOT marking the pair as flat yet.")
                    else:
                        # Both legs confirmed closed (or were already flat).
                        reason = state["close_reason"]
                        exit_z = state["close_z"]
                        a_dir = "short" if direction == "short_a_long_b" else "long"
                        b_dir = "long" if direction == "short_a_long_b" else "short"
                        pnl_a = leg_pnl(a_dir, state["entry_price_a"], price_a, state["qty_a"])
                        pnl_b = leg_pnl(b_dir, state["entry_price_b"], price_b, state["qty_b"])
                        pnl = pnl_a + pnl_b

                        print_pair_close(
                            direction, state["entry_price_a"], price_a,
                            state["entry_price_b"], price_b, pnl, exit_z, reason,
                        )
                        log_trade_event({
                            "event": "close", "direction": direction,
                            "entry_price_a": state["entry_price_a"], "exit_price_a": price_a,
                            "entry_price_b": state["entry_price_b"], "exit_price_b": price_b,
                            "pnl": pnl, "z": exit_z,
                            "reason": "stopped_out" if reason == "stopped out" else "target_hit",
                        })
                        trades_taken += 1
                        total_pnl += pnl

                        if reason == "stopped out":
                            state["cooldown_until"] = now + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)

                        state["direction"] = None
                        state["closing"] = False
                        state["close_reason"] = None
                        state["close_z"] = None

        except Exception as e:
            log(f"ERROR processing pair this cycle: {e}. Will retry next cycle.")

        time.sleep(POLL_INTERVAL_SECONDS)

    # End of day: close anything still open.
    if state["direction"] is not None:
        try:
            price_a = get_latest_price(SYMBOL_A)
            price_b = get_latest_price(SYMBOL_B)
            closed_a, closed_b = close_pair_with_retries()
            if price_a is not None and price_b is not None and closed_a is not False and closed_b is not False:
                direction = state["direction"]
                a_dir = "short" if direction == "short_a_long_b" else "long"
                b_dir = "long" if direction == "short_a_long_b" else "short"
                pnl_a = leg_pnl(a_dir, state["entry_price_a"], price_a, state["qty_a"])
                pnl_b = leg_pnl(b_dir, state["entry_price_b"], price_b, state["qty_b"])
                pnl = pnl_a + pnl_b
                final_z = (price_a / price_b - ratio_mean) / ratio_stdev if ratio_mean and ratio_stdev else float("nan")
                print_pair_close(
                    direction, state["entry_price_a"], price_a,
                    state["entry_price_b"], price_b, pnl, final_z, "end of day",
                )
                log_trade_event({
                    "event": "close", "direction": direction,
                    "entry_price_a": state["entry_price_a"], "exit_price_a": price_a,
                    "entry_price_b": state["entry_price_b"], "exit_price_b": price_b,
                    "pnl": pnl, "reason": "end_of_day",
                })
                trades_taken += 1
                total_pnl += pnl
            else:
                log("CRITICAL: end-of-day close did not fully succeed - see prior CRITICAL lines. "
                    "This day's P&L total below does NOT include this unresolved position.")
        except Exception as e:
            log(f"ERROR closing pair at end of day: {e}. Check both legs manually on Alpaca.")

    print_day_summary(day, total_pnl, trades_taken)
    log_trade_event({"event": "day_summary", "date": str(day), "pnl": total_pnl, "trades": trades_taken})


def wait_until(target_dt):
    while True:
        now = datetime.now(ET)
        if now >= target_dt:
            return
        remaining = (target_dt - now).total_seconds()
        time.sleep(min(remaining, 30))


def main():
    print_startup_banner()
    log(f"Pairs live bot starting. Paper mode: {PAPER}. Pair: {SYMBOL_A} / {SYMBOL_B}.")
    while True:
        try:
            clock = trading_client.get_clock()
        except Exception as e:
            log(f"ERROR checking market clock: {e}. Retrying in 30 seconds.")
            time.sleep(30)
            continue

        if not clock.is_open:
            next_open = clock.next_open.astimezone(ET)
            log(f"Market closed. Sleeping until next open at {next_open}.")
            wait_until(next_open)
            continue

        today = datetime.now(ET).date()
        run_trading_day(today)

        try:
            clock = trading_client.get_clock()
            next_open = clock.next_open.astimezone(ET)
        except Exception as e:
            log(f"ERROR checking market clock after trading day: {e}. Retrying in 30 seconds.")
            time.sleep(30)
            continue
        wait_until(next_open)


def close_all_positions():
    for symbol in (SYMBOL_A, SYMBOL_B):
        try:
            close_leg(symbol)
        except Exception as e:
            log(f"Could not close {symbol} on shutdown: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user. Attempting to close any open positions before exiting.")
        close_all_positions()
        log("Exiting.")
    except Exception as e:
        log(f"FATAL ERROR: {e}. Attempting to close any open positions before exiting.")
        close_all_positions()
        raise
