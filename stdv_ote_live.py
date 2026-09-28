"""
STDV + OTE with HTF PD Arrays - LIVE (paper) multi-symbol trading bot for
Alpaca.

Places REAL orders against your Alpaca account (paper by default). Own
symbols (XLE, XLU), own log file, runs alongside your other bots.

IMPORTANT HONESTY NOTE: genuine ICT-style "HTF PD arrays" (order blocks,
fair value gaps) are identified by eye from chart structure, not by a
fixed formula - there's no way to code "what a trader would circle on a
chart" without just picking arbitrary rules and calling it that. This
bot is a systematic, OBJECTIVE PROXY for the same underlying idea, built
from things that actually compute:

  - HTF PD array proxy: today's premium/discount split is just the
    midpoint (equilibrium) of the last HTF_LOOKBACK_DAYS of daily
    high/low range. Above midpoint = premium, below = discount.
  - OTE (Optimal Trade Entry): the real Fibonacci 61.8%-79% retracement
    zone of TODAY's intraday swing (from the session's high/low so far).
  - STDV: an actual standard deviation of recent 1-min closes, used to
    size the stop-loss distance (a real statistical measure, not a
    decorative label).

Strategy per symbol:
- Figure out which extreme (today's high or today's low) was set more
  recently. That tells you the direction of the current intraday swing:
  low-then-high = swing UP (impulse up), high-then-low = swing DOWN.
- Swing UP -> wait for a pullback DOWN into the 61.8%-79% retracement
  zone measured from the high back toward the low. If that zone also
  sits BELOW the HTF equilibrium (a "discount"), go LONG - the idea
  being: buying a fair-value pullback in an uptrend, at a discounted
  price relative to the broader range.
- Swing DOWN -> mirror image: pullback UP into the 61.8%-79% zone, and
  if that sits ABOVE HTF equilibrium (a "premium"), go SHORT.
- Confidence-based sizing ($100-$500): how deep into the 61.8%-79% zone
  price actually is (79% = higher confidence than 62%).
- Stop-loss = entry +/- HARD_STOP_STDEV_MULT * (stdev of recent 1-min
  closes) - a real volatility-scaled stop, not a fixed distance.
- Once a first profit checkpoint is reached, the stop trails behind the
  best price instead of taking a fixed profit - same mechanic as the
  other trend-oriented bots (orb_live.py, crt_live.py).
- One position per symbol at a time. Cooldown after any stop-out.
- Exit at 3:55pm ET if still open.

Requires:
    pip install alpaca-py pytz colorama pandas

    export APCA_API_KEY_ID="your_key"
    export APCA_API_SECRET_KEY="your_secret"

Run:
    python stdv_ote_live.py

Every trade (open and close) is appended to stdv_ote_trade_log.jsonl in
the same folder - one JSON object per line.

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
# vwap: QQQ, IWM / vp: BTC/USD / pairs: VOO, IVV / crt: XLF, XLK).
SYMBOLS = [
    {"symbol": "XLE", "asset_class": "stock"},
    {"symbol": "XLU", "asset_class": "stock"},
]

BASE_NOTIONAL = 100.0        # trade size at confidence == 0
MAX_NOTIONAL = 500.0         # trade size at confidence == 1 (maxed out)
# Confidence = how deep into the 61.8%-79% OTE zone price is when the
# trade triggers, clamped to [0, 1].

HTF_LOOKBACK_DAYS = 10        # calendar days back for the HTF premium/discount range
OTE_LOW = 0.618               # OTE zone starts at this retracement fraction
OTE_HIGH = 0.79                # OTE zone ends at this retracement fraction
STDV_LOOKBACK_BARS = 30      # 1-min bars used to compute the stop-sizing stdev
MIN_BARS_FOR_STATS = 30      # need at least this many bars before trusting swing/stdev data
RECALC_SECONDS = 60          # how often to refetch bars and recompute swing/stdev

TAKE_PROFIT_RANGE_MULT = 1.0   # first profit checkpoint = entry +/- this * today's swing width
MIN_TARGET_FRACTION = -0.3     # shrinking-target floor, same mechanic as the other bots
HARD_STOP_STDEV_MULT = 2.0     # stop = entry +/- this * stdev(recent 1-min closes)
TRAIL_LOCK_FRACTION = 0.5      # trailing-stop lock fraction, same mechanic as orb_live.py

MAX_DAILY_LOSS_USD = 100.0
REENTRY_COOLDOWN_MINUTES = 10
POLL_INTERVAL_SECONDS = 15
PAPER = True
LOG_FILE = "stdv_ote_trade_log.jsonl"

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
    confidence = 0.0 if scale <= 0 else max(0.0, min(1.0, extra / scale))
    notional = BASE_NOTIONAL + confidence * (MAX_NOTIONAL - BASE_NOTIONAL)
    return confidence, notional


def shrinking_target_price(entry, direction, target_distance, worst_adverse):
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
    title = "S T D V + O T E   T R A D E R"
    subtitle = f"{', '.join(s['symbol'] for s in SYMBOLS)}  ·  HTF discount/premium OTE"
    _box([title, subtitle], Fore.CYAN, double=True)


def print_trade_open(symbol, direction, price, size_desc):
    _box(
        [f"OPENED {direction.upper()} - {symbol} (OTE entry)", f"  Entry: ~{price:.2f}  Size: {size_desc}"],
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
            f"END OF DAY SUMMARY (STDV+OTE) - {day}",
            f"  Trades taken: {trades_taken}",
            f"  Total P&L: {sign}${total_pnl:.2f}",
        ],
        color,
    )


# ---------------- DATA / ORDER HELPERS ----------------

def get_htf_equilibrium(sym_cfg, day):
    """Once-per-day HTF premium/discount midpoint from the last
    HTF_LOOKBACK_DAYS of daily bars. Returns None if there's no data."""
    symbol = sym_cfg["symbol"]
    start = ET.localize(datetime.combine(day - timedelta(days=HTF_LOOKBACK_DAYS * 2), MARKET_OPEN))
    end = ET.localize(datetime.combine(day, MARKET_OPEN))
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX)
    bars = stock_data_client.get_stock_bars(req).df
    if bars.empty:
        return None
    bars = bars.tail(HTF_LOOKBACK_DAYS)
    htf_high = float(bars["high"].max())
    htf_low = float(bars["low"].min())
    return (htf_high + htf_low) / 2.0


def get_intraday_bars(sym_cfg, day, now):
    symbol = sym_cfg["symbol"]
    start = ET.localize(datetime.combine(day, MARKET_OPEN))
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=now, feed=DataFeed.IEX)
    return stock_data_client.get_stock_bars(req).df


def compute_swing_and_stdev(sym_cfg, day, now):
    """Returns (day_high, day_low, impulse_direction, stdev) or
    (None, None, None, None) if there isn't enough data yet.
    impulse_direction is 'up' if the low happened before the high
    (swing currently up), 'down' if the high happened first."""
    try:
        bars = get_intraday_bars(sym_cfg, day, now)
    except Exception as e:
        log(f"WARNING: could not fetch intraday bars for {sym_cfg['symbol']}: {e}")
        return None, None, None, None

    if bars is None or bars.empty or len(bars) < MIN_BARS_FOR_STATS:
        return None, None, None, None

    bars = bars.reset_index()
    high_pos = bars["high"].values.argmax()
    low_pos = bars["low"].values.argmin()
    if high_pos == low_pos:
        return None, None, None, None

    day_high = float(bars["high"].max())
    day_low = float(bars["low"].min())
    impulse_direction = "up" if low_pos < high_pos else "down"

    stdev = float(bars["close"].tail(STDV_LOOKBACK_BARS).std(ddof=0))
    if stdev <= 0:
        return None, None, None, None

    return day_high, day_low, impulse_direction, stdev


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
    log(f"--- Starting STDV+OTE trading day {day} ---")

    states = {}
    for sym_cfg in SYMBOLS:
        symbol = sym_cfg["symbol"]
        equilibrium = None
        for attempt in range(5):
            try:
                equilibrium = get_htf_equilibrium(sym_cfg, day)
                break
            except Exception as e:
                log(f"ERROR fetching HTF equilibrium for {symbol} (attempt {attempt + 1}/5): {e}. Retrying in 15s.")
                time.sleep(15)

        if equilibrium is None:
            log(f"No HTF data for {symbol} today. Skipping this symbol.")
            continue

        log(f"HTF equilibrium for {symbol}: {equilibrium:.2f}")
        states[symbol] = {
            "cfg": sym_cfg,
            "equilibrium": equilibrium,
            "day_high": None,
            "day_low": None,
            "impulse_direction": None,
            "stdev": None,
            "next_recalc": None,
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

                if state["next_recalc"] is None or now >= state["next_recalc"]:
                    day_high, day_low, impulse_direction, stdev = compute_swing_and_stdev(sym_cfg, day, now)
                    state["day_high"], state["day_low"] = day_high, day_low
                    state["impulse_direction"], state["stdev"] = impulse_direction, stdev
                    state["next_recalc"] = now + timedelta(seconds=RECALC_SECONDS)

                if state["day_high"] is None:
                    continue  # not enough data yet today

                price = get_latest_price(sym_cfg)
                if price is None:
                    continue

                day_high, day_low = state["day_high"], state["day_low"]
                range_width = day_high - day_low
                equilibrium = state["equilibrium"]
                stdev = state["stdev"]

                if state["direction"] is None:
                    if loss_limit_hit:
                        continue
                    if state["cooldown_until"] is not None:
                        if now < state["cooldown_until"]:
                            continue
                        state["cooldown_until"] = None

                    if range_width <= 0:
                        continue

                    if state["impulse_direction"] == "up":
                        # Pullback down into the OTE zone measured from the high back toward the low.
                        zone_low = day_high - OTE_HIGH * range_width
                        zone_high = day_high - OTE_LOW * range_width
                        in_discount = price < equilibrium
                        if zone_low <= price <= zone_high and in_discount:
                            retracement = (day_high - price) / range_width
                            confidence, notional = confidence_and_notional(retracement - OTE_LOW, OTE_HIGH - OTE_LOW)
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
                                state["hard_stop"] = price - HARD_STOP_STDEV_MULT * stdev
                                state["initial_hard_stop"] = state["hard_stop"]
                                state["best_price"] = price
                                state["trailing_active"] = False
                                state["worst_adverse"] = 0.0
                                print_trade_open(symbol, "long", price, size_description(qty, notional, confidence))
                                log_trade_event({
                                    "event": "open", "symbol": symbol, "direction": "long", "price": price,
                                    "qty": qty, "notional": notional, "confidence": confidence,
                                })

                    elif state["impulse_direction"] == "down":
                        # Pullback up into the OTE zone measured from the low back toward the high.
                        zone_low = day_low + OTE_LOW * range_width
                        zone_high = day_low + OTE_HIGH * range_width
                        in_premium = price > equilibrium
                        if zone_low <= price <= zone_high and in_premium:
                            retracement = (price - day_low) / range_width
                            confidence, notional = confidence_and_notional(retracement - OTE_LOW, OTE_HIGH - OTE_LOW)
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
                                state["hard_stop"] = price + HARD_STOP_STDEV_MULT * stdev
                                state["initial_hard_stop"] = state["hard_stop"]
                                state["best_price"] = price
                                state["trailing_active"] = False
                                state["worst_adverse"] = 0.0
                                print_trade_open(symbol, "short", price, size_description(qty, notional, confidence))
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
                        state["cooldown_until"] = now + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
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
    log(f"STDV+OTE live bot starting. Paper mode: {PAPER}. Symbols: {[s['symbol'] for s in SYMBOLS]}.")
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
