"""
Volume Profile Value-Area Breakout - LIVE (paper) multi-symbol trading bot
for Alpaca.

Places REAL orders against your Alpaca account (paper by default).
Meant to run ALONGSIDE orb_live.py and vwap_live.py, not instead of them -
different logic, its own log file, its own positions.

Strategy per symbol:
- Before the trading day starts, build YESTERDAY's volume profile from
  minute bars: bin traded price into PROFILE_BINS buckets, sum volume per
  bucket. Find the Point of Control (POC, the bucket with the most
  volume), then grow a "value area" out from the POC bucket, adding
  whichever neighboring bucket (above or below) has more volume, until
  VALUE_AREA_PCT of the day's total volume is enclosed. The top/bottom
  edges of that region are Value Area High (VAH) and Value Area Low (VAL).
- Break above VAH -> buy. Break below VAL -> sell short.
- Take-profit target starts at entry +/- TAKE_PROFIT_VA_MULT * (value area
  width, VAH - VAL). While the position is underwater, the target shrinks
  toward (and can go slightly past) breakeven based on the worst
  unrealized loss seen so far -- see shrinking_target_price().
- One open position per symbol at a time. After any close, that symbol is
  on cooldown before it can re-enter.
- Exit at 3:55pm ET if still open.

v2 - EXPERIMENTAL RISK CHANGE: there is no stop-loss in this version. A
losing position is only closed when its take-profit target -- which
shrinks the deeper the position goes underwater -- gets caught up to by
price, or when the end-of-day forced close happens. That end-of-day close
is the only remaining hard backstop on how far a loss can run. This is
meant for watching how the mechanic behaves in paper trading, not as
something to run with real money.

Requires:
    pip install alpaca-py pytz colorama

    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python vp_live.py

Every trade (open and close) is appended to vp_trade_log.jsonl in the
same folder - one JSON object per line.

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
SYMBOLS = [
    {"symbol": "BTC/USD", "asset_class": "crypto"},
]

BASE_NOTIONAL = 100.0        # trade size at confidence == 0
MAX_NOTIONAL = 500.0         # trade size at confidence == 1 (maxed out)
# Confidence = how far past VAH/VAL the price already was at the moment of
# the breakout, as a fraction of the value area's own width (VAH - VAL),
# clamped to [0, 1]. Simple proxy, not a validated predictor - check it
# against your own trade history once you've got enough logged.

PROFILE_BINS = 50            # how many price buckets to split yesterday's range into
VALUE_AREA_PCT = 0.68        # fraction of yesterday's volume the value area should hold
LOOKBACK_DAYS = 5            # how many calendar days back to search for the last session's bars
TAKE_PROFIT_VA_MULT = 1.0    # initial target = entry +/- this * (VAH - VAL)
MIN_TARGET_FRACTION = -0.3   # once fully shrunk, will accept exiting at a small loss
                              # (this fraction of the original target distance) rather
                              # than shrinking forever
HARD_STOP_MULT = 2.0         # real stop-loss = entry +/- this * (VAH - VAL). Wider than the
                              # old opposite-edge stop so normal chop doesn't trigger it, but
                              # it DOES cap max loss regardless of whether price ever recovers.
MAX_DAILY_LOSS_USD = 100.0
REENTRY_COOLDOWN_MINUTES = 20  # wait this long after any close before re-entering that symbol
POLL_INTERVAL_SECONDS = 15
PAPER = True
LOG_FILE = "vp_trade_log.jsonl"

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
    title = "V P   T R A D E R"
    subtitle = f"{', '.join(s['symbol'] for s in SYMBOLS)}  ·  value-area breakout"
    _box([title, subtitle], Fore.CYAN, double=True)


def print_trade_open(symbol, direction, price, size_desc):
    _box(
        [f"OPENED {direction.upper()} - {symbol} (VP breakout)", f"  Entry: ~{price:.2f}  Size: {size_desc}"],
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
            f"END OF DAY SUMMARY (VP) - {day}",
            f"  Trades taken: {trades_taken}",
            f"  Total P&L: {sign}${total_pnl:.2f}",
        ],
        color,
    )


# ---------------- DATA / ORDER HELPERS ----------------

def get_previous_session_bars(sym_cfg, day):
    """Fetch minute bars for the most recent trading session before `day`."""
    symbol = sym_cfg["symbol"]
    search_start = ET.localize(datetime.combine(day - timedelta(days=LOOKBACK_DAYS), MARKET_OPEN))
    search_end = ET.localize(datetime.combine(day, MARKET_OPEN))

    if sym_cfg["asset_class"] == "stock":
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
            start=search_start, end=search_end, feed=DataFeed.IEX,
        )
        bars = stock_data_client.get_stock_bars(req).df
    else:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
            start=search_start, end=search_end,
        )
        bars = crypto_data_client.get_crypto_bars(req).df

    if bars.empty:
        return None

    # Keep only the most recent calendar day present in the data.
    bars = bars.reset_index()
    ts_col = "timestamp" if "timestamp" in bars.columns else bars.columns[1]
    bars["_date"] = bars[ts_col].dt.tz_convert(ET).dt.date
    last_date = bars["_date"].max()
    return bars[bars["_date"] == last_date]


def compute_value_area(bars):
    """Returns (poc, vah, val) or (None, None, None) if there's not enough data."""
    if bars is None or bars.empty:
        return None, None, None

    price_low = float(bars["low"].min())
    price_high = float(bars["high"].max())
    if price_high <= price_low:
        return None, None, None

    bin_width = (price_high - price_low) / PROFILE_BINS
    bin_volume = [0.0] * PROFILE_BINS

    for _, row in bars.iterrows():
        typical_price = (row["high"] + row["low"] + row["close"]) / 3.0
        idx = int((typical_price - price_low) / bin_width)
        idx = max(0, min(PROFILE_BINS - 1, idx))
        bin_volume[idx] += float(row["volume"])

    total_volume = sum(bin_volume)
    if total_volume <= 0:
        return None, None, None

    poc_idx = bin_volume.index(max(bin_volume))

    included = {poc_idx}
    accumulated = bin_volume[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx

    while accumulated < VALUE_AREA_PCT * total_volume:
        next_low_idx = low_idx - 1
        next_high_idx = high_idx + 1
        vol_low = bin_volume[next_low_idx] if next_low_idx >= 0 else -1
        vol_high = bin_volume[next_high_idx] if next_high_idx < PROFILE_BINS else -1

        if vol_low < 0 and vol_high < 0:
            break  # ran out of bins on both sides

        if vol_high >= vol_low:
            high_idx = next_high_idx
            accumulated += bin_volume[high_idx]
        else:
            low_idx = next_low_idx
            accumulated += bin_volume[low_idx]

    poc = price_low + (poc_idx + 0.5) * bin_width
    val = price_low + low_idx * bin_width
    vah = price_low + (high_idx + 1) * bin_width
    return poc, vah, val


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

    At worst_adverse == 0: target == the original full target. At
    worst_adverse == -target_distance: target == entry (breakeven).
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
    log(f"--- Starting Volume Profile trading day {day} ---")

    states = {}
    for sym_cfg in SYMBOLS:
        symbol = sym_cfg["symbol"]
        poc, vah, val = None, None, None
        for attempt in range(5):
            try:
                bars = get_previous_session_bars(sym_cfg, day)
                poc, vah, val = compute_value_area(bars)
                break
            except Exception as e:
                log(f"ERROR building volume profile for {symbol} (attempt {attempt + 1}/5): {e}. Retrying in 15s.")
                time.sleep(15)

        if vah is None:
            log(f"No volume profile data for {symbol} today. Skipping this symbol.")
            continue

        log(f"Value area for {symbol}: VAH={vah:.2f}, POC={poc:.2f}, VAL={val:.2f}")
        states[symbol] = {
            "cfg": sym_cfg,
            "vah": vah,
            "val": val,
            "poc": poc,
            "direction": None,
            "entry_price": None,
            "qty": None,
            "notional": None,
            "confidence": None,
            "target_distance": None,
            "hard_stop": None,
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
                        continue

                    if state["cooldown_until"] is not None:
                        if datetime.now(ET) < state["cooldown_until"]:
                            continue
                        state["cooldown_until"] = None

                    if price > state["vah"]:
                        va_width = state["vah"] - state["val"]
                        confidence, notional = confidence_and_notional(price - state["vah"], va_width)
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
                        state["target_distance"] = TAKE_PROFIT_VA_MULT * va_width
                        state["hard_stop"] = price - HARD_STOP_MULT * va_width
                        state["worst_adverse"] = 0.0
                        size_desc = size_description(sym_cfg["asset_class"], qty, notional, confidence, price)
                        print_trade_open(symbol, "long", price, size_desc)
                        log_trade_event({
                            "event": "open", "symbol": symbol, "direction": "long", "price": price,
                            "qty": qty, "notional": notional, "confidence": confidence,
                        })

                    elif price < state["val"]:
                        if sym_cfg["asset_class"] == "crypto":
                            continue  # Alpaca doesn't support short-selling crypto
                        va_width = state["vah"] - state["val"]
                        confidence, notional = confidence_and_notional(state["val"] - price, va_width)
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
                        state["target_distance"] = TAKE_PROFIT_VA_MULT * va_width
                        state["hard_stop"] = price + HARD_STOP_MULT * va_width
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

                    unrealized = (price - entry) if direction == "long" else (entry - price)
                    if unrealized < state["worst_adverse"]:
                        state["worst_adverse"] = unrealized

                    target_price = shrinking_target_price(
                        entry, direction, state["target_distance"], state["worst_adverse"]
                    )
                    hit_target = (direction == "long" and price >= target_price) or \
                                 (direction == "short" and price <= target_price)
                    hit_hard_stop = (direction == "long" and price <= state["hard_stop"]) or \
                                    (direction == "short" and price >= state["hard_stop"])

                    if hit_target or hit_hard_stop:
                        closed_qty = close_position(symbol)
                        state["direction"] = None
                        state["cooldown_until"] = datetime.now(ET) + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
                        if closed_qty is not None:
                            size = state["qty"] if sym_cfg["asset_class"] == "stock" else state["notional"] / entry
                            pnl = unrealized * size
                            reason = "stopped out" if hit_hard_stop else "target hit"
                            print_trade_close(symbol, direction, entry, price, pnl, reason)
                            log_trade_event({
                                "event": "close", "symbol": symbol, "direction": direction,
                                "entry_price": entry, "exit_price": price, "pnl": pnl,
                                "reason": "stopped_out" if hit_hard_stop else "target_hit",
                                "target_price": target_price, "worst_adverse_per_unit": state["worst_adverse"],
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
    log(f"Volume Profile live bot starting. Paper mode: {PAPER}. Symbols: {[s['symbol'] for s in SYMBOLS]}.")
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
