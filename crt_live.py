"""
CRT (Candle Range Theory) - LIVE (paper) multi-symbol trading bot for Alpaca.

Places REAL orders against your Alpaca account (paper by default).
Meant to run ALONGSIDE your other bots - different logic, its own log
file, its own symbols (XLF, XLK - not used by any other bot you're
running, so positions never collide on the same symbol).

Strategy per symbol - this IS the actual core idea behind "Candle Range
Theory": yesterday's full session range (high to low) is the reference
candle. Today, if price sweeps BELOW yesterday's low and then closes
back above it, that's read as a liquidity grab / stop-hunt, not a real
breakdown - go LONG, targeting yesterday's high. Mirror image for a
sweep above yesterday's high that reclaims back below it -> SHORT,
targeting yesterday's low. This is a reversal strategy, not a breakout
one - opposite psychology from orb_live.py and vp_live.py.

- Confidence-based position sizing ($100-$500): how deep the sweep went
  beyond the level, as a fraction of yesterday's range width. A shallow
  poke past the level is low confidence; a deep sweep is higher
  confidence. Simple proxy, not a validated predictor - check it against
  your own trade history once you've got enough logged.
- Real stop-loss beyond the sweep's extreme (HARD_STOP_MULT * range
  width past the sweep low/high) - caps loss regardless of what price
  does next.
- Once price reaches a first profit checkpoint, the stop starts trailing
  behind the best price reached instead of taking a fixed profit -
  same "let a real reversal keep running" mechanic as orb_live.py.
- One open position per symbol at a time. Cooldown after any stop-out.
- Exit at 3:55pm ET if still open.

Requires:
    pip install alpaca-py pytz colorama

    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python crt_live.py

Every trade (open and close) is appended to crt_trade_log.jsonl in the
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
# Must not overlap with symbols any other bot trades (orb: SPY, DIA /
# vwap: QQQ, IWM / vp: BTC/USD / pairs: VOO, IVV). See the note in
# pairs_live.py about why overlap causes real problems.
SYMBOLS = [
    {"symbol": "XLF", "asset_class": "stock"},
    {"symbol": "XLK", "asset_class": "stock"},
]

BASE_NOTIONAL = 100.0        # trade size at confidence == 0
MAX_NOTIONAL = 500.0         # trade size at confidence == 1 (maxed out)
# Confidence = how deep the sweep went past yesterday's high/low, as a
# fraction of yesterday's own range width, clamped to [0, 1].

LOOKBACK_DAYS = 5            # how many calendar days back to search for the last session's bars
TAKE_PROFIT_RANGE_MULT = 0.5  # first profit checkpoint = entry +/- this * yesterday's range width.
                               # Reaching it starts trailing instead of closing (see below).
MIN_TARGET_FRACTION = -0.3    # once fully shrunk on the downside, will accept exiting at a
                               # small loss (this fraction of the original target distance)
HARD_STOP_MULT = 0.5          # stop = sweep extreme +/- this * yesterday's range width
TRAIL_LOCK_FRACTION = 0.5     # once trailing, stop trails this fraction of the original
                               # entry-to-hard-stop distance behind the best price seen
MAX_DAILY_LOSS_USD = 100.0
REENTRY_COOLDOWN_MINUTES = 10
POLL_INTERVAL_SECONDS = 15
PAPER = True
LOG_FILE = "crt_trade_log.jsonl"

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


def confidence_and_notional(extra, scale):
    """extra/scale, clamped to [0, 1], mapped linearly onto [BASE_NOTIONAL, MAX_NOTIONAL]."""
    confidence = 0.0 if scale <= 0 else max(0.0, min(1.0, extra / scale))
    notional = BASE_NOTIONAL + confidence * (MAX_NOTIONAL - BASE_NOTIONAL)
    return confidence, notional


def shrinking_target_price(entry, direction, target_distance, worst_adverse):
    """Same mechanic as the other bots: the profit target shrinks toward
    (and can go slightly past) entry as the trade goes further against
    you first, so a trade that was briefly red can still exit near
    breakeven instead of needing a full round trip to the original
    target. Floors at MIN_TARGET_FRACTION * target_distance."""
    if target_distance <= 0:
        return entry
    shrink_fraction = max(MIN_TARGET_FRACTION, 1.0 + worst_adverse / target_distance)
    current_distance = target_distance * shrink_fraction
    return entry + current_distance if direction == "long" else entry - current_distance


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
    tl, tr, bl, br, h, v = ("╔", "╗", "╚", "╝", "═", "║") if double else ("┌", "┐", "└", "┘", "─", "│")
    width = max(len(line) for line in lines) + 2
    print(color + tl + h * width + tr)
    for line in lines:
        print(color + v + " " + line.ljust(width - 2) + " " + v)
    print(color + bl + h * width + br + Style.RESET_ALL)


def print_startup_banner():
    title = "C R T   T R A D E R"
    subtitle = f"{', '.join(s['symbol'] for s in SYMBOLS)}  ·  sweep & reclaim"
    _box([title, subtitle], Fore.CYAN, double=True)


def print_trade_open(symbol, direction, price, size_desc):
    _box(
        [f"OPENED {direction.upper()} - {symbol} (CRT sweep/reclaim)", f"  Entry: ~{price:.2f}  Size: {size_desc}"],
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
            f"END OF DAY SUMMARY (CRT) - {day}",
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

    req = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
        start=search_start, end=search_end, feed=DataFeed.IEX,
    )
    bars = stock_data_client.get_stock_bars(req).df
    if bars.empty:
        return None

    bars = bars.reset_index()
    ts_col = "timestamp" if "timestamp" in bars.columns else bars.columns[1]
    bars["_date"] = bars[ts_col].dt.tz_convert(ET).dt.date
    last_date = bars["_date"].max()
    return bars[bars["_date"] == last_date]


def get_latest_price(sym_cfg):
    symbol = sym_cfg["symbol"]
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


def place_entry_order(symbol, side, qty):
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


def close_position(symbol):
    qty = get_open_position_qty(symbol)
    if qty == 0:
        return None
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    order = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY)
    trading_client.submit_order(order)
    log(f"Closed position of {qty} in {symbol}.")
    return qty


def size_description(qty, notional, confidence):
    return f"{qty} shares (confidence {confidence * 100:.0f}%)"


# ---------------- CORE TRADING LOGIC ----------------

def run_trading_day(day):
    log(f"--- Starting CRT trading day {day} ---")

    states = {}
    for sym_cfg in SYMBOLS:
        symbol = sym_cfg["symbol"]
        prev_high = prev_low = None
        for attempt in range(5):
            try:
                bars = get_previous_session_bars(sym_cfg, day)
                if bars is not None and not bars.empty:
                    prev_high = float(bars["high"].max())
                    prev_low = float(bars["low"].min())
                break
            except Exception as e:
                log(f"ERROR fetching previous session bars for {symbol} (attempt {attempt + 1}/5): {e}. Retrying in 15s.")
                time.sleep(15)

        if prev_high is None:
            log(f"No previous-session data for {symbol} today. Skipping this symbol.")
            continue

        log(f"Previous session range for {symbol}: high={prev_high:.2f}, low={prev_low:.2f}")
        states[symbol] = {
            "cfg": sym_cfg,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "swept_below": False,
            "sweep_low_extreme": None,
            "swept_above": False,
            "sweep_high_extreme": None,
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
        now = datetime.now(ET)
        loss_limit_hit = total_pnl <= -MAX_DAILY_LOSS_USD

        for symbol, state in states.items():
            try:
                sym_cfg = state["cfg"]
                price = get_latest_price(sym_cfg)
                if price is None:
                    continue

                if state["direction"] is None:
                    range_width = state["prev_high"] - state["prev_low"]

                    # Track sweeps below the low and above the high.
                    if price < state["prev_low"]:
                        state["swept_below"] = True
                        low_ext = state["sweep_low_extreme"]
                        state["sweep_low_extreme"] = price if low_ext is None else min(low_ext, price)
                    if price > state["prev_high"]:
                        state["swept_above"] = True
                        high_ext = state["sweep_high_extreme"]
                        state["sweep_high_extreme"] = price if high_ext is None else max(high_ext, price)

                    if loss_limit_hit:
                        continue
                    if state["cooldown_until"] is not None:
                        if now < state["cooldown_until"]:
                            continue
                        state["cooldown_until"] = None

                    # Reclaim from below -> long.
                    if state["swept_below"] and price > state["prev_low"]:
                        sweep_depth = state["prev_low"] - state["sweep_low_extreme"]
                        confidence, notional = confidence_and_notional(sweep_depth, range_width)
                        qty = max(1, round(notional / price))
                        try:
                            place_entry_order(symbol, OrderSide.BUY, qty)
                        except Exception as e:
                            log(f"WARNING: entry order failed for {symbol} (long): {e}")
                        else:
                            state["direction"] = "long"
                            state["entry_price"] = price
                            state["qty"] = qty
                            state["notional"] = notional
                            state["confidence"] = confidence
                            state["target_distance"] = TAKE_PROFIT_RANGE_MULT * range_width
                            state["hard_stop"] = state["sweep_low_extreme"] - HARD_STOP_MULT * range_width
                            state["initial_hard_stop"] = state["hard_stop"]
                            state["best_price"] = price
                            state["trailing_active"] = False
                            state["worst_adverse"] = 0.0
                            print_trade_open(symbol, "long", price, size_description(qty, notional, confidence))
                            log_trade_event({
                                "event": "open", "symbol": symbol, "direction": "long", "price": price,
                                "qty": qty, "notional": notional, "confidence": confidence,
                            })
                        state["swept_below"] = False
                        state["sweep_low_extreme"] = None

                    # Reclaim from above -> short.
                    elif state["swept_above"] and price < state["prev_high"]:
                        sweep_depth = state["sweep_high_extreme"] - state["prev_high"]
                        confidence, notional = confidence_and_notional(sweep_depth, range_width)
                        qty = max(1, round(notional / price))
                        try:
                            place_entry_order(symbol, OrderSide.SELL, qty)
                        except Exception as e:
                            log(f"WARNING: entry order failed for {symbol} (short): {e}")
                        else:
                            state["direction"] = "short"
                            state["entry_price"] = price
                            state["qty"] = qty
                            state["notional"] = notional
                            state["confidence"] = confidence
                            state["target_distance"] = TAKE_PROFIT_RANGE_MULT * range_width
                            state["hard_stop"] = state["sweep_high_extreme"] + HARD_STOP_MULT * range_width
                            state["initial_hard_stop"] = state["hard_stop"]
                            state["best_price"] = price
                            state["trailing_active"] = False
                            state["worst_adverse"] = 0.0
                            print_trade_open(symbol, "short", price, size_description(qty, notional, confidence))
                            log_trade_event({
                                "event": "open", "symbol": symbol, "direction": "short", "price": price,
                                "qty": qty, "notional": notional, "confidence": confidence,
                            })
                        state["swept_above"] = False
                        state["sweep_high_extreme"] = None

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
                            pnl = unrealized * state["qty"]
                            reason = "trailing stop" if was_trailing else "stopped out"
                            print_trade_close(symbol, direction, entry, price, pnl, reason)
                            log_trade_event({
                                "event": "close", "symbol": symbol, "direction": direction,
                                "entry_price": entry, "exit_price": price, "pnl": pnl,
                                "reason": "trailing_stop" if was_trailing else "stopped_out",
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
            price = get_latest_price(state["cfg"])
            closed_qty = close_position(symbol)
            if price is not None and closed_qty is not None:
                direction = state["direction"]
                entry = state["entry_price"]
                pnl_per_unit = (price - entry) if direction == "long" else (entry - price)
                pnl = pnl_per_unit * state["qty"]
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
    log(f"CRT live bot starting. Paper mode: {PAPER}. Symbols: {[s['symbol'] for s in SYMBOLS]}.")
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
