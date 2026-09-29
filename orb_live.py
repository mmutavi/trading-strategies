"""
Opening Range Breakout (ORB) - LIVE (paper) multi-symbol trading bot for Alpaca.

Places REAL orders against your Alpaca account (paper by default).
Read the whole file before running it. Do not set PAPER = False against a
live-money account until you've watched this behave correctly for a while.

v2 - EXPERIMENTAL RISK CHANGE: there is no stop-loss in this version. A
losing position is only closed when its take-profit target -- which
shrinks the deeper the position goes underwater -- gets caught up to by
price, or when the end-of-day forced close happens. That end-of-day close
is the only remaining hard backstop on how far a loss can run. This is
meant for watching how the mechanic behaves in paper trading, not as
something to run with real money.

Strategy per symbol:
- Opening range = high/low of the first N minutes after 9:30am ET.
- Break above OR high -> buy. Break below OR low -> sell short.
- Take-profit target starts at entry +/- TAKE_PROFIT_RANGE_MULT * (opening
  range width). While the position is underwater, the target shrinks
  toward (and can go slightly past) breakeven based on the worst
  unrealized loss seen so far -- see shrinking_target_price().
- Exit at 3:55pm ET if still open (applies to crypto too, for simplicity -
  crypto actually trades 24/7, this just keeps everything on one clock).
- One trade per symbol per day, max. Running several symbols is how you get
  more than one trade a day without abandoning the ORB logic on any single one.

Requires:
    pip install alpaca-py pytz colorama

    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python orb_live.py

Every trade (open and close) is appended to trade_log.jsonl in the same
folder - one JSON object per line. That file is meant to be readable by
some future dashboard/app without needing to parse terminal output.

Stop the bot any time with Ctrl+C - it will try to close any open
positions before exiting.
"""

import os
import json
import time
import math
from datetime import datetime, time as dtime, timedelta
import pytz

from colorama import init as colorama_init, Fore, Style

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

colorama_init(autoreset=True)

# ---------------- CONFIG ----------------
# asset_class: "stock" or "crypto"
SYMBOLS = [
    {"symbol": "SPY", "asset_class": "stock"},
    {"symbol": "DIA", "asset_class": "stock"},
]

BASE_NOTIONAL = 100.0            # trade size at confidence == 0
MAX_NOTIONAL = 500.0             # trade size at confidence == 1 (maxed out)
# Confidence = how far price broke beyond the opening range, as a fraction of
# the range's own width, clamped to [0, 1]. A breakout that clears the range
# by a full range-width again is "maxed out" confidence; a breakout that
# barely ticks past the line is confidence ~0. This is a simple proxy, not a
# validated predictor - worth checking against your own trade history once
# you've got enough of them logged to see if bigger breakouts actually do
# better.

OPENING_RANGE_MINUTES = 5        # narrower range -> more breakout signals -> more trades
TAKE_PROFIT_RANGE_MULT = 1.0     # first profit checkpoint = entry +/- this * opening range width.
                                  # Reaching it doesn't close the trade - it starts trailing (see below).
MIN_TARGET_FRACTION = -0.3       # once fully shrunk, will accept exiting at a small loss
                                  # (this fraction of the original target distance) rather
                                  # than shrinking forever
HARD_STOP_MULT = 2.0             # real stop-loss = entry +/- this * opening range width.
                                  # Wider than the old "opposite side of the range" stop so
                                  # normal chop doesn't trigger it, but it DOES cap max loss
                                  # per trade regardless of whether price ever comes back.
TRAIL_LOCK_FRACTION = 0.5        # once the profit checkpoint is reached, the stop trails behind
                                  # the best price seen by this fraction of the ORIGINAL entry-to-
                                  # hard-stop distance, only ever moving in your favor. Lets a real
                                  # trend keep running instead of capping it at a fixed target.
MAX_DAILY_LOSS_USD = 100.0       # bot stops opening NEW trades for the rest of the day past this
REENTRY_COOLDOWN_MINUTES = 10    # wait this long after any close before re-entering that symbol
POLL_INTERVAL_SECONDS = 15
PAPER = True                     # keep True until you deliberately decide otherwise
LOG_FILE = "trade_log.jsonl"

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
crypto_data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)


def confidence_and_notional(extra, scale):
    """extra/scale, clamped to [0, 1], mapped linearly onto [BASE_NOTIONAL, MAX_NOTIONAL]."""
    confidence = 0.0 if scale <= 0 else max(0.0, min(1.0, extra / scale))
    notional = BASE_NOTIONAL + confidence * (MAX_NOTIONAL - BASE_NOTIONAL)
    return confidence, notional


# ---------------- LOGGING / DISPLAY ----------------

def log(msg):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_trade_event(record):
    """Append a structured trade event to the JSONL log file for a future dashboard."""
    record["timestamp"] = datetime.now(ET).isoformat()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log(f"WARNING: could not write to {LOG_FILE}: {e}")


def _box(lines, color, double=True):
    """Draw a clean unicode box around the given lines (no emoji, no
    rainbow - just crisp single/double-line borders). Shared style across
    every bot for a consistent look."""
    tl, tr, bl, br, h, v = ("╔", "╗", "╚", "╝", "═", "║") if double else ("┌", "┐", "└", "┘", "─", "│")
    width = max(len(line) for line in lines) + 2
    print(color + tl + h * width + tr)
    for line in lines:
        print(color + v + " " + line.ljust(width - 2) + " " + v)
    print(color + bl + h * width + br + Style.RESET_ALL)


def print_startup_banner():
    title = "O R B   T R A D E R"
    subtitle = f"{', '.join(s['symbol'] for s in SYMBOLS)}  ·  opening range breakout"
    _box([title, subtitle], Fore.CYAN, double=True)


def print_trade_open(symbol, direction, price, size_desc):
    _box(
        [f"OPENED {direction.upper()} - {symbol}", f"  Entry: ~{price:.2f}  Size: {size_desc}"],
        Fore.CYAN,
    )


def print_trade_close(symbol, direction, entry_price, exit_price, pnl, reason):
    color = Fore.GREEN if pnl >= 0 else Fore.RED
    sign = "+" if pnl >= 0 else ""
    _box(
        [
            f"CLOSED {direction.upper()} - {symbol} ({reason})",
            f"  Entry: {entry_price:.2f}  Exit: {exit_price:.2f}",
            f"  P&L: {sign}${pnl:.2f}",
        ],
        color,
    )


def print_day_summary(day, total_pnl, trades_taken):
    color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    sign = "+" if total_pnl >= 0 else ""
    _box(
        [
            f"END OF DAY SUMMARY - {day}",
            f"  Trades taken: {trades_taken}",
            f"  Total P&L: {sign}${total_pnl:.2f}",
        ],
        color,
    )


# ---------------- DATA / ORDER HELPERS ----------------

def get_opening_range(sym_cfg, day):
    symbol = sym_cfg["symbol"]
    start = ET.localize(datetime.combine(day, MARKET_OPEN))
    end = start + timedelta(minutes=OPENING_RANGE_MINUTES)

    if sym_cfg["asset_class"] == "stock":
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end, feed=DataFeed.IEX
        )
        bars = stock_data_client.get_stock_bars(req).df
    else:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end)
        bars = crypto_data_client.get_crypto_bars(req).df

    if bars.empty:
        return None, None
    return float(bars["high"].max()), float(bars["low"].min())


def get_latest_price(sym_cfg):
    symbol = sym_cfg["symbol"]
    try:
        if sym_cfg["asset_class"] == "stock":
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            quote = stock_data_client.get_stock_latest_quote(req)[symbol]
        else:
            req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = crypto_data_client.get_crypto_latest_quote(req)[symbol]
    except Exception as e:
        log(f"WARNING: could not fetch quote for {symbol}: {e}")
        return None

    if quote.bid_price is None or quote.ask_price is None:
        return None
    if quote.bid_price <= 0 or quote.ask_price <= 0:
        return None
    return (quote.bid_price + quote.ask_price) / 2


def place_entry_order(symbol, asset_class, side, qty=None, notional=None):
    if asset_class == "stock":
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    else:
        order = MarketOrderRequest(symbol=symbol, notional=notional, side=side, time_in_force=TimeInForce.GTC)
    return trading_client.submit_order(order)


def position_lookup_symbol(symbol):
    """Alpaca's position endpoint wants crypto symbols without the slash
    (e.g. 'BTCUSD'), even though orders use 'BTC/USD'. Stock symbols are
    unaffected since they never contain a slash."""
    return symbol.replace("/", "")


def get_open_position_qty(symbol):
    lookup_symbol = position_lookup_symbol(symbol)
    try:
        position = trading_client.get_open_position(lookup_symbol)
        return float(position.qty)
    except Exception as e:
        msg = str(e).lower()
        if "position does not exist" not in msg and "404" not in msg and "not found" not in msg:
            log(f"WARNING: unexpected error checking position for {symbol} ({e}). Treating as no position.")
        return 0.0


def close_position(symbol):
    qty = get_open_position_qty(symbol)
    if qty == 0:
        return None
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    time_in_force = TimeInForce.DAY if "/" not in symbol else TimeInForce.GTC
    order = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=side, time_in_force=time_in_force)
    trading_client.submit_order(order)
    log(f"Closed position of {qty} in {symbol}.")
    return qty


def size_description(asset_class, qty, notional, confidence, price):
    base = f"{qty:g} shares (~${qty * price:.2f} notional)" if asset_class == "stock" else f"${notional:.2f} notional"
    return f"{base} (confidence {confidence * 100:.0f}%)"


def shrinking_target_price(entry, direction, target_distance, worst_adverse):
    """The take-profit level for this instant. worst_adverse is the most
    negative unrealized P&L-per-unit seen so far for this trade (0.0 if
    it's never gone underwater) -- it only ever gets more negative over
    the life of the trade, so this target only ever shrinks, never
    recovers back toward the original level.

    At worst_adverse == 0: target == the original full target.
    At worst_adverse == -target_distance: target == entry (breakeven).
    Beyond that it keeps shrinking until MIN_TARGET_FRACTION, then holds
    there rather than shrinking indefinitely.
    """
    if target_distance <= 0:
        return entry
    shrink_fraction = 1.0 + (worst_adverse / target_distance)
    shrink_fraction = max(MIN_TARGET_FRACTION, min(1.0, shrink_fraction))
    current_distance = target_distance * shrink_fraction
    return entry + current_distance if direction == "long" else entry - current_distance


# ---------------- CORE TRADING LOGIC ----------------

def run_trading_day(day):
    log(f"--- Starting trading day {day} ---")

    or_end = ET.localize(datetime.combine(day, MARKET_OPEN)) + timedelta(minutes=OPENING_RANGE_MINUTES)
    wait_until(or_end)

    states = {}
    for sym_cfg in SYMBOLS:
        symbol = sym_cfg["symbol"]
        or_high, or_low = None, None
        for attempt in range(5):
            try:
                or_high, or_low = get_opening_range(sym_cfg, day)
                break
            except Exception as e:
                log(f"ERROR fetching opening range for {symbol} (attempt {attempt + 1}/5): {e}. Retrying in 15s.")
                time.sleep(15)

        if or_high is None:
            log(f"No opening range data for {symbol} today. Skipping this symbol.")
            continue

        log(f"Opening range for {symbol}: high={or_high:.2f}, low={or_low:.2f}")
        states[symbol] = {
            "cfg": sym_cfg,
            "or_high": or_high,
            "or_low": or_low,
            "direction": None,
            "entry_price": None,
            "qty": None,
            "notional": None,
            "confidence": None,
            "target_distance": None,
            "hard_stop": None,
            "initial_hard_stop": None,
            "best_price": None,
            "trailing_active": False,
            "worst_adverse": 0.0,
            "cooldown_until": None,
        }

    exit_dt = ET.localize(datetime.combine(day, EXIT_TIME))
    trades_taken = 0
    total_pnl = 0.0

    while datetime.now(ET) < exit_dt:
        loss_limit_hit = total_pnl <= -MAX_DAILY_LOSS_USD

        for symbol, state in states.items():
            try:
                sym_cfg = state["cfg"]
                price = get_latest_price(sym_cfg)
                if price is None:
                    continue

                if state["direction"] is None:
                    if loss_limit_hit:
                        continue  # no new entries once the daily loss cap is hit

                    if state["cooldown_until"] is not None:
                        if datetime.now(ET) < state["cooldown_until"]:
                            continue  # still cooling down after a stop-out
                        state["cooldown_until"] = None

                    if price > state["or_high"]:
                        or_width = state["or_high"] - state["or_low"]
                        breakout_distance = price - state["or_high"]
                        confidence, notional = confidence_and_notional(breakout_distance, or_width)
                        qty = round(notional / price, 6) if sym_cfg["asset_class"] == "stock" else None
                        if sym_cfg["asset_class"] == "stock":
                            notional = qty * price
                        try:
                            place_entry_order(
                                symbol, sym_cfg["asset_class"], OrderSide.BUY,
                                qty=qty, notional=notional if sym_cfg["asset_class"] == "crypto" else None,
                            )
                        except Exception as e:
                            log(f"WARNING: entry order failed for {symbol} (long): {e}")
                            continue
                        state["direction"] = "long"
                        state["entry_price"] = price
                        state["qty"] = qty
                        state["notional"] = notional
                        state["confidence"] = confidence
                        state["target_distance"] = TAKE_PROFIT_RANGE_MULT * or_width
                        state["hard_stop"] = price - HARD_STOP_MULT * or_width
                        state["initial_hard_stop"] = state["hard_stop"]
                        state["best_price"] = price
                        state["trailing_active"] = False
                        state["worst_adverse"] = 0.0
                        size_desc = size_description(sym_cfg["asset_class"], qty, notional, confidence, price)
                        print_trade_open(symbol, "long", price, size_desc)
                        log_trade_event({
                            "event": "open", "symbol": symbol, "direction": "long", "price": price,
                            "qty": qty, "notional": notional, "confidence": confidence,
                        })

                    elif price < state["or_low"]:
                        if sym_cfg["asset_class"] == "crypto":
                            # Alpaca doesn't support short-selling crypto. Skip this
                            # side entirely rather than send an order that will fail.
                            continue
                        or_width = state["or_high"] - state["or_low"]
                        breakout_distance = state["or_low"] - price
                        confidence, notional = confidence_and_notional(breakout_distance, or_width)
                        qty = math.floor(notional / price)
                        if qty < 1:
                            log(f"Skipping {symbol} short: one share at ${price:.2f} exceeds the ${notional:.2f} confidence size.")
                            continue
                        notional = qty * price
                        try:
                            place_entry_order(symbol, sym_cfg["asset_class"], OrderSide.SELL, qty=qty)
                        except Exception as e:
                            log(f"WARNING: entry order failed for {symbol} (short): {e}")
                            continue
                        state["direction"] = "short"
                        state["entry_price"] = price
                        state["qty"] = qty
                        state["notional"] = notional
                        state["confidence"] = confidence
                        state["target_distance"] = TAKE_PROFIT_RANGE_MULT * or_width
                        state["hard_stop"] = price + HARD_STOP_MULT * or_width
                        state["initial_hard_stop"] = state["hard_stop"]
                        state["best_price"] = price
                        state["trailing_active"] = False
                        state["worst_adverse"] = 0.0
                        size_desc = size_description(sym_cfg["asset_class"], qty, notional, confidence, price)
                        print_trade_open(symbol, "short", price, size_desc)
                        log_trade_event({
                            "event": "open", "symbol": symbol, "direction": "short", "price": price,
                            "qty": qty, "notional": notional, "confidence": confidence,
                        })
                else:
                    direction = state["direction"]
                    entry = state["entry_price"]

                    if direction == "long":
                        state["best_price"] = max(state["best_price"], price)
                    else:
                        state["best_price"] = min(state["best_price"], price)

                    unrealized = (price - entry) if direction == "long" else (entry - price)
                    if unrealized < state["worst_adverse"]:
                        state["worst_adverse"] = unrealized

                    target_price = shrinking_target_price(
                        entry, direction, state["target_distance"], state["worst_adverse"]
                    )

                    if not state["trailing_active"]:
                        reached_target = (direction == "long" and price >= target_price) or \
                                          (direction == "short" and price <= target_price)
                        if reached_target:
                            state["trailing_active"] = True

                    if state["trailing_active"]:
                        risk_distance = abs(entry - state["initial_hard_stop"])
                        if direction == "long":
                            trail_stop = state["best_price"] - TRAIL_LOCK_FRACTION * risk_distance
                            state["hard_stop"] = max(state["hard_stop"], trail_stop)
                        else:
                            trail_stop = state["best_price"] + TRAIL_LOCK_FRACTION * risk_distance
                            state["hard_stop"] = min(state["hard_stop"], trail_stop)

                    hit_hard_stop = (direction == "long" and price <= state["hard_stop"]) or \
                                    (direction == "short" and price >= state["hard_stop"])

                    if hit_hard_stop:
                        closed_qty = close_position(symbol)
                        was_trailing = state["trailing_active"]
                        state["direction"] = None
                        state["cooldown_until"] = datetime.now(ET) + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
                        if closed_qty is not None:
                            size = state["qty"] if sym_cfg["asset_class"] == "stock" else state["notional"] / entry
                            pnl = unrealized * size
                            reason = "trailing stop" if was_trailing else "stopped out"
                            print_trade_close(symbol, direction, entry, price, pnl, reason)
                            log_trade_event({
                                "event": "close", "symbol": symbol, "direction": direction,
                                "entry_price": entry, "exit_price": price, "pnl": pnl,
                                "reason": "trailing_stop" if was_trailing else "stopped_out",
                                "worst_adverse_per_unit": state["worst_adverse"],
                            })
                            trades_taken += 1
                            total_pnl += pnl
                        else:
                            log(f"WARNING: expected an open position to close for {symbol} but found none.")
            except Exception as e:
                log(f"ERROR processing {symbol} this cycle: {e}. Will retry next cycle.")

        time.sleep(POLL_INTERVAL_SECONDS)

    # End of day: close anything still open.
    for symbol, state in states.items():
        if state["direction"] is None:
            continue
        try:
            sym_cfg = state["cfg"]
            price = get_latest_price(sym_cfg)
            closed_qty = close_position(symbol)
            if price is not None and closed_qty is not None:
                direction = state["direction"]
                entry = state["entry_price"]
                pnl_per_unit = (price - entry) if direction == "long" else (entry - price)
                size = state["qty"] if sym_cfg["asset_class"] == "stock" else state["notional"] / entry
                pnl = pnl_per_unit * size
                print_trade_close(symbol, direction, entry, price, pnl, "end of day")
                log_trade_event({
                    "event": "close", "symbol": symbol, "direction": direction,
                    "entry_price": entry, "exit_price": price, "pnl": pnl, "reason": "end_of_day",
                })
                trades_taken += 1
                total_pnl += pnl
        except Exception as e:
            log(f"ERROR closing {symbol} at end of day: {e}. Check this position manually on Alpaca.")

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
    log(f"ORB live bot starting. Paper mode: {PAPER}. Symbols: {[s['symbol'] for s in SYMBOLS]}.")
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
    for sym_cfg in SYMBOLS:
        try:
            close_position(sym_cfg["symbol"])
        except Exception as e:
            log(f"Could not close {sym_cfg['symbol']} on shutdown: {e}")


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
