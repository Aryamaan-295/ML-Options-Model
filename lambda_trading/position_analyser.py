import os
import math
import traceback
import boto3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient

# ==========================================
# HYPERPARAMETERS (must mirror scanner)
# ==========================================
IST              = timezone(timedelta(hours=5, minutes=30))
DEFAULT_LOT_SIZE = 75
ENTRY_TIME_STR   = "09:20:00"   # options spread snapshot time
EXIT_TIME_STR    = "15:30:00"   # options spread snapshot time (last tick)
# The 5-min OHLCV collection stamps candles by their OPEN time.
# The last candle of the trading day covers 15:25–15:30 and is labelled 15:25.
INDEX_ENTRY_TIME = "09:20:00"   # 5-min OHLCV candle label for entry spot
INDEX_EXIT_TIME  = "15:25:00"   # 5-min OHLCV candle label for exit spot (last candle)

ssm = boto3.client('ssm', region_name="ap-south-1")

# ==========================================
# HELPERS
# ==========================================
def get_secret(parameter_name):
    try:
        res = ssm.get_parameter(Name=f'/fyers/{parameter_name}', WithDecryption=True)
        return res['Parameter']['Value']
    except Exception as e:
        print(f"Failed to fetch {parameter_name}: {e}")
        return None

def send_telegram_alert(message):
    bot_token = get_secret('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id   = get_secret('TELEGRAM_CHAT_ID')   or os.getenv('TELEGRAM_CHAT_ID')
    if not bot_token or not chat_id:
        print("Telegram Alert (Not Sent):\n", message)
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"Telegram Error: {e}")

def _parse_expiry_to_str(exp_str) -> str | None:
    """Converts any Fyers expiry string to 'YYYY-MM-DD' for consistent matching."""
    exp_str = str(exp_str).strip().upper()
    try:
        if len(exp_str) == 5 and exp_str[2:].isalpha():
            dt = pd.to_datetime(exp_str, format="%y%b")
            return dt.strftime('%Y-%m-%d')
        elif len(exp_str) >= 5:
            yy, m, dd = exp_str[:2], exp_str[2], exp_str[3:5]
            month_map = {
                '1': '01', '2': '02', '3': '03', '4': '04', '5': '05',
                '6': '06', '7': '07', '8': '08', '9': '09',
                'O': '10', 'N': '11', 'D': '12'
            }
            mm = month_map.get(m, '01')
            dt = pd.to_datetime(f"{yy}{mm}{dd}", format="%y%m%d")
            return dt.strftime('%Y-%m-%d')
    except Exception:
        pass
    return None

def _build_snapshot_df(spreads_col, datetime_str: str, fallback_window: tuple = None) -> tuple[pd.DataFrame, str]:
    """
    Fetches a snapshot from nifty_spreads at the given datetime_str.
    If not found and fallback_window=(start, end) is provided, uses the
    latest available row in that window instead.
    Returns (DataFrame, actual_datetime_used).
    """
    rows = list(spreads_col.find({"datetime": datetime_str}))
    actual_dt = datetime_str

    if not rows and fallback_window:
        start, end = fallback_window
        fallback = spreads_col.find_one(
            {"datetime": {"$gte": start, "$lte": end}},
            sort=[("datetime", -1)]
        )
        if fallback:
            actual_dt = fallback['datetime']
            rows = list(spreads_col.find({"datetime": actual_dt}))

    if not rows:
        return pd.DataFrame(), actual_dt

    df = pd.DataFrame(rows)
    for col in ['strike', 'bid', 'ask', 'volume', 'ltp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()
    df = (df.sort_values('datetime')
            .groupby(['strike', 'type', 'expiry'])
            .tail(1)
            .reset_index(drop=True))

    df.rename(columns={'type': 'Type', 'strike': 'Strike'}, inplace=True)
    df['expiry_str'] = df['expiry'].apply(_parse_expiry_to_str)
    df.dropna(subset=['expiry_str'], inplace=True)

    return df, actual_dt


def _resolve_index_spot(index_col, today_str: str, target_time: str,
                        fallback_window: tuple) -> tuple[float | None, str | None]:
    """
    Looks up the close price from the nifty_5m_ohlcv collection for the
    candle at `target_time`. Falls back to the latest candle within
    `fallback_window` if the exact timestamp is missing.

    Returns (close_price, actual_datetime_used) or (None, None) on failure.

    FIX: Previously the code used `index_col.find_one({"datetime": actual_exit_dt})`
    where actual_exit_dt was "15:30:00" — a timestamp that never exists in the
    5-min OHLCV collection (last candle is labelled "15:25:00"). On miss it
    silently fell through to `exit_df['Strike'].median()`, producing a bogus
    spot value (~24,000 in the example report).
    """
    target_dt = f"{today_str} {target_time}"
    row = index_col.find_one({"datetime": target_dt})
    if row:
        return float(row['close']), target_dt

    # Fallback: latest candle within the window
    start, end = fallback_window
    row = index_col.find_one(
        {"datetime": {"$gte": start, "$lte": end}},
        sort=[("datetime", -1)]
    )
    if row:
        print(f"Index spot: exact candle {target_dt!r} not found; "
              f"using fallback {row['datetime']!r} (close={row['close']})")
        return float(row['close']), row['datetime']

    print(f"Index spot: NO candle found between {start} and {end}.")
    return None, None


# ==========================================
# CHARGES
# ==========================================
def calculate_charges(legs: list, price_key: str, side: str) -> float:
    """
    Generic charge calculator.

    side='entry':
        BUY legs  → buy side  (stamp duty)
        SELL legs → sell side (STT)

    side='exit':
        Original BUY legs are now SOLD  → sell side (STT)
        Original SELL legs are bought back → buy side (stamp duty)
    """
    if side == 'entry':
        buy_legs  = [l for l in legs if l['action'] == 'buy']
        sell_legs = [l for l in legs if l['action'] == 'sell']
    else:  # exit — directions reversed
        buy_legs  = [l for l in legs if l['action'] == 'sell']  # buying back shorts
        sell_legs = [l for l in legs if l['action'] == 'buy']   # selling longs

    buy_to  = sum(abs(l[price_key] * l['qty'] * l['lot_size']) for l in buy_legs)
    sell_to = sum(abs(l[price_key] * l['qty'] * l['lot_size']) for l in sell_legs)
    total   = buy_to + sell_to

    brokerage = 20 * len(legs)
    stt       = sell_to * 0.001
    exc_txn   = total   * 0.0003503
    sebi      = total   * 0.000001
    stamp     = buy_to  * 0.00003
    ipft      = total   * 0.000005
    gst       = 0.18 * (brokerage + exc_txn + sebi)
    return brokerage + stt + exc_txn + sebi + stamp + ipft + gst

# ==========================================
# PRICE RESOLUTION
# ==========================================
def resolve_prices(legs: list, snapshot_df: pd.DataFrame,
                   price_key: str, direction: str) -> tuple[list, list]:
    """
    Looks up each leg in snapshot_df and records the fill price.

    direction='entry': BUY legs fill at ASK, SELL legs fill at BID.
    direction='exit':  BUY legs fill at BID (selling back), SELL legs fill at ASK (buying back).

    Writes the price into each leg dict under `price_key`.
    Returns (enriched_legs, warnings).
    """
    enriched, warnings = [], []

    for leg in legs:
        strike     = float(leg['strike'])
        opt_type   = leg['type'].upper()
        expiry_str = leg['expiry_str']

        mask = (
            (snapshot_df['Strike']     == strike)   &
            (snapshot_df['Type']       == opt_type) &
            (snapshot_df['expiry_str'] == expiry_str)
        )
        row = snapshot_df[mask]
        enriched_leg = dict(leg)

        if row.empty:
            warnings.append(f"{opt_type}{int(strike)} ({expiry_str}): not found in {direction} snapshot")
            enriched_leg[price_key]          = None
            enriched_leg[f'{direction}_bid'] = None
            enriched_leg[f'{direction}_ask'] = None
        else:
            row = row.iloc[0]
            bid = float(row.get('bid', 0))
            ask = float(row.get('ask', 0))

            if direction == 'entry':
                fill = ask if leg['action'] == 'buy' else bid
            else:  # exit
                fill = bid if leg['action'] == 'buy' else ask

            enriched_leg[price_key]          = fill
            enriched_leg[f'{direction}_bid'] = bid
            enriched_leg[f'{direction}_ask'] = ask

        enriched.append(enriched_leg)

    return enriched, warnings

# ==========================================
# TELEGRAM MESSAGE BUILDER
# ==========================================
def _format_exit_block(doc: dict) -> str:
    label     = doc['strategy_label']
    expiry    = doc['expiry']
    entry_m   = doc['metrics_at_entry']
    legs      = doc['legs']
    exit_spot = doc.get('exit_spot', 0)

    rpnl_gross    = doc.get('realized_pnl_gross')
    entry_charges = doc.get('entry_charges')
    exit_charges  = doc.get('exit_charges')
    rpnl_net      = doc.get('realized_pnl_net')

    if rpnl_net is None:
        return f"**{label}** | `Could not calculate — missing prices`\n------"

    sign          = "+" if rpnl_net >= 0 else ""
    outcome_emoji = "✅" if rpnl_net >= 0 else "❌"
    total_charges = (entry_charges or 0) + (exit_charges or 0)

    lines = [
        f"{outcome_emoji} **{label}**",
        f"Expiry: `{expiry}` | Exit Spot: Rs `{exit_spot:,.2f}`",
        "",
        "Realized PnL:",
        f"- Gross PnL:       Rs `{sign}{rpnl_gross:,.2f}`",
        f"- Entry Charges:   Rs `{entry_charges:,.2f}`",
        f"- Exit Charges:    Rs `{exit_charges:,.2f}`",
        f"- Total Charges:   Rs `{total_charges:,.2f}`",
        f"- **Net PnL:       Rs `{sign}{rpnl_net:,.2f}`**",
        "",
        f"vs Theoretical:    Max `{entry_m['Max_Profit_1D']:,.2f}` / Min `{entry_m['Min_Profit_1D']:,.2f}`",
        "",
        "Leg Detail (9:20 Entry → 15:30 Exit):",
    ]

    for leg in legs:
        action_str    = "BUY " if leg['action'] == 'buy' else "SELL"
        actual_entry  = leg.get('actual_entry_price')
        exit_p        = leg.get('exit_price')
        leg_pnl       = leg.get('leg_realized_pnl')
        scanner_entry = leg.get('entry_price')  # stored from 9:15 scan (reference only)

        entry_str = f"Rs `{actual_entry:.2f}`" if actual_entry is not None else "`N/A`"
        exit_str  = f"Rs `{exit_p:.2f}`"       if exit_p       is not None else "`N/A`"
        pnl_str   = (f"PnL: Rs `{'+' if leg_pnl >= 0 else ''}{leg_pnl:,.2f}`"
                     if leg_pnl is not None else "PnL: `N/A`")
        ref_str   = f" *(scan: {scanner_entry:.2f})*" if scanner_entry else ""

        lines.append(
            f"- {action_str} {leg['type']} {int(leg['strike'])} | "
            f"Entry{ref_str}: {entry_str} → Exit: {exit_str} | {pnl_str}"
        )

    lines.append("------")
    return "\n".join(lines)

# ==========================================
# LAMBDA HANDLER
# ==========================================
def lambda_handler(event, context):
    try:
        now_ist   = datetime.now(IST)
        today_str = now_ist.strftime('%Y-%m-%d')

        entry_datetime_str = f"{today_str} {ENTRY_TIME_STR}"
        exit_datetime_str  = f"{today_str} {EXIT_TIME_STR}"

        mongo_uri = get_secret('MONGO_URI') or os.getenv('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI not found.")

        client      = MongoClient(mongo_uri)
        baskets_col = client['deployments']['option_baskets']
        spreads_col = client['trading_data']['nifty_spreads']
        index_col   = client['trading_data']['nifty_5m_ohlcv']

        # ── 1. Fetch today's open baskets ─────────────────────────────────────
        open_baskets = list(baskets_col.find({
            'scan_date': today_str,
            'status':    'open',
        }))

        if not open_baskets:
            msg = f"Exit PnL Scanner: No open baskets found for `{today_str}`. Nothing to do."
            print(msg)
            send_telegram_alert(msg)
            return {"statusCode": 200, "body": msg}

        print(f"Found {len(open_baskets)} open baskets for {today_str}.")

        # ── 2. Load 9:20am entry snapshot (options spreads) ──────────────────
        entry_df, actual_entry_dt = _build_snapshot_df(
            spreads_col,
            entry_datetime_str,
            fallback_window=(f"{today_str} 09:15:00", f"{today_str} 09:25:00")
        )

        if entry_df.empty:
            msg = f"Exit PnL Scanner: No entry snapshot found near 09:20 for `{today_str}`."
            print(msg)
            send_telegram_alert(msg)
            return {"statusCode": 200, "body": msg}

        print(f"Entry snapshot resolved to: {actual_entry_dt}")

        # ── 3. Load 3:30pm exit snapshot (options spreads) ───────────────────
        exit_df, actual_exit_dt = _build_snapshot_df(
            spreads_col,
            exit_datetime_str,
            fallback_window=(f"{today_str} 15:00:00", f"{today_str} 15:35:00")
        )

        if exit_df.empty:
            msg = f"Exit PnL Scanner: No exit snapshot found near 15:30 for `{today_str}`."
            print(msg)
            send_telegram_alert(msg)
            return {"statusCode": 200, "body": msg}

        print(f"Exit snapshot resolved to: {actual_exit_dt}")

        # ── 4. Fetch entry and exit spot prices from 5-min OHLCV ─────────────
        # FIX: Use dedicated _resolve_index_spot() with the correct candle labels.
        # The exit spot must come from the 15:25 candle (last 5-min candle of the
        # day, covering 15:25–15:30). The old code queried "15:30:00" which never
        # exists, silently fell back to Strike.median(), and produced a bogus value.
        entry_spot, actual_entry_idx_dt = _resolve_index_spot(
            index_col, today_str,
            target_time=INDEX_ENTRY_TIME,
            fallback_window=(f"{today_str} 09:15:00", f"{today_str} 09:25:00")
        )
        exit_spot, actual_exit_idx_dt = _resolve_index_spot(
            index_col, today_str,
            target_time=INDEX_EXIT_TIME,           # "15:25:00" — last 5-min candle
            fallback_window=(f"{today_str} 15:20:00", f"{today_str} 15:30:00")
        )

        # Hard-fail if either spot is unresolvable — don't proceed with garbage data
        if entry_spot is None:
            msg = (f"Exit PnL Scanner: Could not resolve entry spot from nifty_5m_ohlcv "
                   f"near {INDEX_ENTRY_TIME} for `{today_str}`. Aborting.")
            print(msg)
            send_telegram_alert(msg)
            return {"statusCode": 200, "body": msg}

        if exit_spot is None:
            msg = (f"Exit PnL Scanner: Could not resolve exit spot from nifty_5m_ohlcv "
                   f"near {INDEX_EXIT_TIME} (15:25 candle) for `{today_str}`. Aborting.")
            print(msg)
            send_telegram_alert(msg)
            return {"statusCode": 200, "body": msg}

        print(f"Entry spot ({actual_entry_idx_dt}): {entry_spot}")
        print(f"Exit spot  ({actual_exit_idx_dt}):  {exit_spot}")

        # ── 5. Compute realized PnL per basket ────────────────────────────────
        all_warnings = []
        updated_docs = []

        for doc in open_baskets:
            legs = doc['legs']

            # — 5a. Resolve actual 9:20am entry prices —
            legs_with_entry, entry_warns = resolve_prices(
                legs, entry_df,
                price_key='actual_entry_price',
                direction='entry'
            )
            if entry_warns:
                all_warnings.append(f"[{doc['strategy_label']}] ENTRY: {'; '.join(entry_warns)}")

            # — 5b. Resolve 3:30pm exit prices —
            legs_with_exit, exit_warns = resolve_prices(
                legs_with_entry, exit_df,
                price_key='exit_price',
                direction='exit'
            )
            if exit_warns:
                all_warnings.append(f"[{doc['strategy_label']}] EXIT: {'; '.join(exit_warns)}")

            # — 5c. Check all prices resolved —
            all_resolved = (
                all(l.get('actual_entry_price') is not None for l in legs_with_exit) and
                all(l.get('exit_price')          is not None for l in legs_with_exit)
            )

            if all_resolved:
                for leg in legs_with_exit:
                    direction_mult = 1 if leg['action'] == 'buy' else -1
                    leg['leg_realized_pnl'] = round(
                        (leg['exit_price'] - leg['actual_entry_price'])
                        * direction_mult
                        * leg['qty']
                        * leg['lot_size'],
                        2
                    )

                gross_pnl     = sum(l['leg_realized_pnl'] for l in legs_with_exit)
                entry_charges = calculate_charges(legs_with_exit, 'actual_entry_price', side='entry')
                exit_charges  = calculate_charges(legs_with_exit, 'exit_price',         side='exit')
                net_pnl       = gross_pnl - entry_charges - exit_charges

                update_payload = {
                    'status':                'closed',
                    'actual_entry_datetime': actual_entry_dt,
                    'actual_entry_idx_dt':   actual_entry_idx_dt,
                    'exit_datetime':         actual_exit_dt,
                    'exit_idx_dt':           actual_exit_idx_dt,
                    'entry_spot':            entry_spot,
                    'exit_spot':             exit_spot,
                    'legs':                  legs_with_exit,
                    'realized_pnl_gross':    round(gross_pnl, 2),
                    'entry_charges':         round(entry_charges, 2),
                    'exit_charges':          round(exit_charges, 2),
                    'realized_pnl_net':      round(net_pnl, 2),
                }
            else:
                for leg in legs_with_exit:
                    ep = leg.get('actual_entry_price')
                    xp = leg.get('exit_price')
                    if ep is not None and xp is not None:
                        direction_mult = 1 if leg['action'] == 'buy' else -1
                        leg['leg_realized_pnl'] = round(
                            (xp - ep) * direction_mult * leg['qty'] * leg['lot_size'], 2
                        )
                    else:
                        leg['leg_realized_pnl'] = None

                update_payload = {
                    'status':                'data_missing',
                    'actual_entry_datetime': actual_entry_dt,
                    'actual_entry_idx_dt':   actual_entry_idx_dt,
                    'exit_datetime':         actual_exit_dt,
                    'exit_idx_dt':           actual_exit_idx_dt,
                    'entry_spot':            entry_spot,
                    'exit_spot':             exit_spot,
                    'legs':                  legs_with_exit,
                    'realized_pnl_gross':    None,
                    'entry_charges':         None,
                    'exit_charges':          None,
                    'realized_pnl_net':      None,
                }

            baskets_col.update_one(
                {'_id': doc['_id']},
                {'$set': update_payload},
            )

            doc.update(update_payload)
            updated_docs.append(doc)

        # ── 6. Build and send Telegram report ─────────────────────────────────
        spot_at_scan = open_baskets[0].get('spot_at_entry', entry_spot)

        header_lines = [
            "**EXIT PnL REPORT**",
            f"Date: `{today_str}`",
            f"Entry Snapshot: `{actual_entry_dt}` | Exit Snapshot: `{actual_exit_dt}`",
            f"Index Candles:  Entry `{actual_entry_idx_dt}` | Exit `{actual_exit_idx_dt}`",
            f"Spot @ Scan (9:15): Rs `{spot_at_scan:,.2f}`",
            f"Spot @ Entry (9:20): Rs `{entry_spot:,.2f}` | Spot @ Exit (15:25 close): Rs `{exit_spot:,.2f}`",
            f"Day Move: `{exit_spot - entry_spot:+,.2f}` pts",
            "------",
        ]

        blocks = [_format_exit_block(doc) for doc in updated_docs]

        if all_warnings:
            blocks.append("⚠️ **Data Warnings:**\n" + "\n".join(f"- {w}" for w in all_warnings))

        final_msg = "\n".join(header_lines) + "\n" + "\n".join(blocks)

        if len(final_msg) > 4000:
            final_msg = final_msg[:3990] + "\n`...(truncated)`"

        print(final_msg)
        send_telegram_alert(final_msg)
        return {"statusCode": 200, "body": f"Exit PnL computed for {len(updated_docs)} baskets."}

    except Exception as e:
        error_trace = traceback.format_exc()
        error_type  = type(e).__name__
        msg         = f"{error_type}: {str(e)}"
        print("CRITICAL ERROR:\n", error_trace)
        alert_msg = f"**Exit PnL Lambda Error:**\n{msg}\n\n**Traceback:**\n`{error_trace[-300:]}`"
        send_telegram_alert(alert_msg)
        return {"statusCode": 500, "body": f"Internal Error: {msg}"}