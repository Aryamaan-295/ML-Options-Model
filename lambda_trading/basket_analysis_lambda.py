import os
import math
import requests
import boto3
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient

# ==========================================
# HYPERPARAMETERS
# ==========================================
IST = timezone(timedelta(hours=5, minutes=30))
MAX_EXPIRY_DAYS       = 35
N_EXPIRIES_TO_CHECK   = 3
WING_WIDTHS           = [50, 100, 150, 200, 250, 300]
DEFAULT_LOT_SIZE      = 75
RISK_FREE_RATE        = 0.065
PAYOFF_SPOTS          = 500
SPOT_SCAN_RADIUS      = 2500

ssm = boto3.client('ssm', region_name="ap-south-1")

# ==========================================
# AWS / UTILITY HELPERS
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

# ==========================================
# BLACK-SCHOLES ENGINE
# ==========================================
_SQRT2     = math.sqrt(2.0)
_SQRT2_INV = 1.0 / _SQRT2

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x * _SQRT2_INV)) / 2.0

def _ncdf_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 + np.frompyfunc(math.erf, 1, 1)(x * _SQRT2_INV).astype(float)) / 2.0

def bs_price_scalar(S: float, K: float, T: float,
                    r: float, sigma: float, opt_type: str) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0) if opt_type == 'CE' else max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = math.exp(-r * T)
    if opt_type == 'CE':
        return S * _ncdf(d1) - K * disc * _ncdf(d2)
    else:
        return K * disc * _ncdf(-d2) - S * _ncdf(-d1)

def bs_price_vec(S_vec: np.ndarray, K: float, T: float,
                 r: float, sigma: float, opt_type: str) -> np.ndarray:
    S_vec = np.asarray(S_vec, dtype=float)
    if T <= 1e-9 or sigma <= 1e-9:
        return np.maximum(S_vec - K, 0.0) if opt_type == 'CE' else np.maximum(K - S_vec, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (np.log(S_vec / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = math.exp(-r * T)
    if opt_type == 'CE':
        return S_vec * _ncdf_vec(d1) - K * disc * _ncdf_vec(d2)
    else:
        return K * disc * _ncdf_vec(-d2) - S_vec * _ncdf_vec(-d1)

def implied_vol_bisect(market_price: float, S: float, K: float, T: float,
                       r: float, opt_type: str, fallback: float = 0.25) -> float:
    if T <= 1e-9 or market_price <= 0:
        return fallback
    intrinsic = max(S - K, 0.0) if opt_type == 'CE' else max(K - S, 0.0)
    if market_price <= intrinsic + 0.01:
        return 0.01
    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        price = bs_price_scalar(S, K, T, r, mid, opt_type)
        if abs(price - market_price) < 0.001:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5

# ==========================================
# OPTIONS DATA UTILITIES
# ==========================================
def parse_fyers_expiry(exp_str):
    exp_str = str(exp_str).strip().upper()
    try:
        if len(exp_str) == 5 and exp_str[2:].isalpha():
            return pd.to_datetime(exp_str, format="%y%b")
        elif len(exp_str) >= 5:
            yy, m, dd = exp_str[:2], exp_str[2], exp_str[3:5]
            month_map = {'1': '01', '2': '02', '3': '03', '4': '04', '5': '05',
                         '6': '06', '7': '07', '8': '08', '9': '09',
                         'O': '10', 'N': '11', 'D': '12'}
            mm = month_map.get(m, '01')
            return pd.to_datetime(f"{yy}{mm}{dd}", format="%y%m%d")
    except Exception:
        pass
    return pd.NaT

def get_leg_data(leg_config: dict, df: pd.DataFrame):
    strike   = float(leg_config['strike'])
    opt_type = str(leg_config['type']).upper().strip()
    expiry   = leg_config['expiry']
    action   = leg_config['action'].lower()

    mask = (df['Expiry'] == expiry) & (df['Strike'] == strike) & (df['Type'] == opt_type)
    row  = df[mask]
    if row.empty:
        return None, "missing"
    row = row.iloc[0]

    entry_price = float(row.get('ask', 0)) if action == 'buy' else float(row.get('bid', 0))
    if entry_price <= 0.0 or float(row.get('Volume', 0)) <= 0.0:
        return None, "illiquid"

    return {
        'symbol':      row['symbol'],
        'strike':      strike,
        'type':        opt_type,
        'action':      action,
        'qty':         leg_config.get('qty', 1),
        'expiry':      expiry,
        'entry_price': entry_price,
        'lot_size':    DEFAULT_LOT_SIZE,
    }, "ok"

def calculate_basket_charges(legs: list) -> float:
    buy_to  = sum(abs(l['cash_premium']) for l in legs if l['action'] == 'buy')
    sell_to = sum(abs(l['cash_premium']) for l in legs if l['action'] == 'sell')
    total   = buy_to + sell_to
    brokerage = 20 * len(legs)
    stt       = sell_to  * 0.001
    exc_txn   = total    * 0.0003503
    sebi      = total    * 0.000001
    stamp     = buy_to   * 0.00003
    ipft      = total    * 0.000005
    gst       = 0.18 * (brokerage + exc_txn + sebi)
    return brokerage + stt + exc_txn + sebi + stamp + ipft + gst

# ==========================================
# 1-DAY PnL CALCULATOR
# ==========================================
def calculate_1d_basket_payoff(portfolio_config: list, df: pd.DataFrame,
                               spot_price: float, buying_date) -> dict:
    legs = []
    for lc in portfolio_config:
        leg_data, status = get_leg_data(lc, df)
        if status != "ok":
            return {"status": status}
        legs.append(leg_data)

    spots = np.sort(np.unique(np.concatenate([
        np.linspace(spot_price - SPOT_SCAN_RADIUS, spot_price + SPOT_SCAN_RADIUS, PAYOFF_SPOTS),
        [l['strike'] for l in legs]
    ]))).astype(float)

    payoff_1d = np.zeros(len(spots), dtype=float)
    total_net_premium = 0.0

    for leg in legs:
        direction = 1 if leg['action'] == 'buy' else -1
        cash_mult = leg['qty'] * leg['lot_size'] * direction

        leg['cash_premium'] = leg['entry_price'] * leg['qty'] * leg['lot_size']
        total_net_premium += leg['cash_premium'] if leg['action'] == 'sell' else -leg['cash_premium']

        exp_date = leg['expiry'].date() if hasattr(leg['expiry'], 'date') else leg['expiry']
        T_entry  = max((exp_date - buying_date).days / 365.0, 1.0 / 252.0)
        T_1d     = max(T_entry - 1.0 / 252.0, 1e-9)

        iv = implied_vol_bisect(leg['entry_price'], spot_price,
                                leg['strike'], T_entry, RISK_FREE_RATE, leg['type'])
        leg['iv'] = iv
        leg['T_entry'] = T_entry

        prices_1d = bs_price_vec(spots, leg['strike'], T_1d, RISK_FREE_RATE, iv, leg['type'])
        payoff_1d += (prices_1d - leg['entry_price']) * cash_mult

    charges = calculate_basket_charges(legs)

    payoff_1d_net = payoff_1d - charges
    max_profit_1d = float(np.max(payoff_1d_net))
    min_profit_1d = float(np.min(payoff_1d_net))

    breakevens = []
    for i in range(len(spots) - 1):
        p0, p1 = payoff_1d_net[i], payoff_1d_net[i + 1]
        if p0 * p1 <= 0 and p0 != p1:
            be = float(spots[i] - p0 * (spots[i + 1] - spots[i]) / (p1 - p0))
            breakevens.append(round(be, 2))

    net_reward = max_profit_1d
    net_risk   = abs(min_profit_1d)
    net_rrr    = net_reward / net_risk if net_risk > 0 else 0.0

    return {
        "status": "success",
        "legs":   legs,
        "metrics": {
            "Max_Profit_1D":    max_profit_1d,
            "Min_Profit_1D":    min_profit_1d,
            "Net_Reward_1D":    net_reward,
            "Net_Risk_1D":      net_risk,
            "Net_RRR_1D":       net_rrr,
            "Total_Premium":    total_net_premium,
            "Charges":          charges,
            "Breakevens":       breakevens,
        }
    }

def _evaluate(portfolio_config, df, spot_price, buying_date) -> dict | None:
    res = calculate_1d_basket_payoff(portfolio_config, df, spot_price, buying_date)
    if res['status'] != 'success':
        return None
    if res['metrics']['Net_Reward_1D'] <= 0 or res['metrics']['Net_Risk_1D'] == 0:
        return None
    return res

# ==========================================
# STRATEGIES
# ==========================================
def scan_iron_condor(df, spot_price, inference, available_expiries, buying_date) -> tuple:
    lo_target = inference['Lower_85pct_CI_Price']
    hi_target = inference['Upper_85pct_CI_Price']

    results, tested = [], 0

    for expiry in available_expiries:
        exp_df = df[df['Expiry'] == expiry]
        pe_strikes_set = set(exp_df[exp_df['Type'] == 'PE']['Strike'].unique())
        ce_strikes_set = set(exp_df[exp_df['Type'] == 'CE']['Strike'].unique())
        pe_strikes = sorted(pe_strikes_set)
        ce_strikes = sorted(ce_strikes_set)

        spe_cands = [s for s in pe_strikes if abs(s - lo_target) <= 400]
        sce_cands = [s for s in ce_strikes if abs(s - hi_target) <= 400]

        for spe in spe_cands:
            for sce in sce_cands:
                if sce <= spe:
                    continue
                for w in WING_WIDTHS:
                    lpe, lce = spe - w, sce + w
                    if lpe in pe_strikes_set and lce in ce_strikes_set:
                        tested += 1
                        portfolio = [
                            {'strike': lpe, 'type': 'PE', 'action': 'buy',  'expiry': expiry, 'qty': 1},
                            {'strike': spe, 'type': 'PE', 'action': 'sell', 'expiry': expiry, 'qty': 1},
                            {'strike': sce, 'type': 'CE', 'action': 'sell', 'expiry': expiry, 'qty': 1},
                            {'strike': lce, 'type': 'CE', 'action': 'buy',  'expiry': expiry, 'qty': 1},
                        ]
                        res = _evaluate(portfolio, df, spot_price, buying_date)
                        if res:
                            m = res['metrics']
                            bes = m['Breakevens']
                            if len(bes) >= 2:
                                be_error = abs(min(bes) - lo_target) + abs(max(bes) - hi_target)
                            else:
                                be_error = float('inf')
                            results.append({
                                'Strategy': 'Iron Condor (85% CI)',
                                'Expiry': str(expiry.date()),
                                'BE_Error': be_error,
                                **m, 'Portfolio': res['legs']
                            })

    results.sort(key=lambda x: (x['BE_Error'], -x['Net_RRR_1D']))
    return results, tested


def scan_long_strangle(df, spot_price, inference, available_expiries, buying_date) -> tuple:
    lo_target = inference['Lower_20pct_CI_Price']
    hi_target = inference['Upper_20pct_CI_Price']

    results, tested = [], 0

    for expiry in available_expiries:
        exp_df = df[df['Expiry'] == expiry]
        pe_strikes = sorted(exp_df[exp_df['Type'] == 'PE']['Strike'].unique())
        ce_strikes = sorted(exp_df[exp_df['Type'] == 'CE']['Strike'].unique())

        pe_cands = [s for s in pe_strikes if spot_price - 1200 <= s <= spot_price]
        ce_cands = [s for s in ce_strikes if spot_price <= s <= spot_price + 1200]

        for pe_s in pe_cands:
            for ce_s in ce_cands:
                tested += 1
                portfolio = [
                    {'strike': pe_s, 'type': 'PE', 'action': 'buy', 'expiry': expiry, 'qty': 1},
                    {'strike': ce_s, 'type': 'CE', 'action': 'buy', 'expiry': expiry, 'qty': 1},
                ]
                res = _evaluate(portfolio, df, spot_price, buying_date)
                if res:
                    m = res['metrics']
                    bes = m['Breakevens']
                    if len(bes) >= 2:
                        be_error = abs(min(bes) - lo_target) + abs(max(bes) - hi_target)
                    else:
                        be_error = float('inf')
                    results.append({
                        'Strategy': 'Long Strangle (20% CI)',
                        'Expiry': str(expiry.date()),
                        'BE_Error': be_error,
                        **m, 'Portfolio': res['legs']
                    })

    results.sort(key=lambda x: (x['BE_Error'], -x['Net_RRR_1D']))
    return results, tested


def scan_directional_spread(df, spot_price, is_bullish, target_short_strike,
                             available_expiries, buying_date) -> tuple:
    """
    Builds a DEBIT vertical spread aimed at the model's predicted target.

    Bullish signal → Bull Call Spread:
        BUY CE at (target_short_strike - wing), SELL CE at target_short_strike.
        The short (sell) strike is placed AT the CI target level so the spread
        is maximally profitable when the market reaches that target.
        1D breakeven sits between the two strikes — sensible, delta-driven.

    Bearish signal → Bear Put Spread:
        BUY PE at (target_short_strike + wing), SELL PE at target_short_strike.
        Same logic: short strike at CI target, profit as market falls toward it.

    Sorting: primary = abs(short_strike - target)  [strike closest to target first]
             secondary = -Net_RRR_1D              [best risk/reward among ties]
    """
    results, tested = [], 0

    for expiry in available_expiries:
        exp_df = df[df['Expiry'] == expiry]

        if is_bullish:
            ce_strikes_set = set(exp_df[exp_df['Type'] == 'CE']['Strike'].unique())
            # Short (sell) leg near the CI target; long (buy) leg = short - wing
            short_cands = sorted(
                s for s in ce_strikes_set if abs(s - target_short_strike) <= 400
            )
            for short_s in short_cands:
                for w in WING_WIDTHS:
                    long_s = short_s - w
                    if long_s in ce_strikes_set:
                        tested += 1
                        portfolio = [
                            {'strike': long_s,  'type': 'CE', 'action': 'buy',  'expiry': expiry, 'qty': 1},
                            {'strike': short_s, 'type': 'CE', 'action': 'sell', 'expiry': expiry, 'qty': 1},
                        ]
                        res = _evaluate(portfolio, df, spot_price, buying_date)
                        if res:
                            be_error = abs(short_s - target_short_strike)
                            results.append({
                                'Expiry':   str(expiry.date()),
                                'BE_Error': be_error,
                                **res['metrics'],
                                'Portfolio': res['legs'],
                            })
        else:
            pe_strikes_set = set(exp_df[exp_df['Type'] == 'PE']['Strike'].unique())
            # Short (sell) leg near the CI target; long (buy) leg = short + wing
            short_cands = sorted(
                s for s in pe_strikes_set if abs(s - target_short_strike) <= 400
            )
            for short_s in short_cands:
                for w in WING_WIDTHS:
                    long_s = short_s + w
                    if long_s in pe_strikes_set:
                        tested += 1
                        portfolio = [
                            {'strike': long_s,  'type': 'PE', 'action': 'buy',  'expiry': expiry, 'qty': 1},
                            {'strike': short_s, 'type': 'PE', 'action': 'sell', 'expiry': expiry, 'qty': 1},
                        ]
                        res = _evaluate(portfolio, df, spot_price, buying_date)
                        if res:
                            be_error = abs(short_s - target_short_strike)
                            results.append({
                                'Expiry':   str(expiry.date()),
                                'BE_Error': be_error,
                                **res['metrics'],
                                'Portfolio': res['legs'],
                            })

    results.sort(key=lambda x: (x['BE_Error'], -x['Net_RRR_1D']))
    return results, tested

# ==========================================
# TELEGRAM MESSAGE BUILDER
# ==========================================
def _format_strategy_block(label: str, strategy_name: str, results: list, tested: int) -> str:
    lines = [f"**{label}**", f"Baskets Evaluated: `{tested}`"]

    if not results:
        lines.append("`No mathematically viable baskets found.`")
        lines.append("------")
        return "\n".join(lines)

    best = results[0]
    be_strs = " & ".join([f"Rs `{b:,.2f}`" for b in best.get('Breakevens', [])]) or "`N/A`"

    lines.extend([
        f"Expiry: `{best['Expiry']}` | Strategy: {strategy_name}",
        "",
        "PnL & Metrics:",
        f"- Max 1D Profit: Rs `{best['Max_Profit_1D']:,.2f}`",
        f"- Max 1D Loss:   Rs `{best['Min_Profit_1D']:,.2f}`",
        f"- Net Reward:    Rs `{best['Net_Reward_1D']:,.2f}`",
        f"- Net Risk:      Rs `{best['Net_Risk_1D']:,.2f}`",
        f"- 1D Net RRR:    `{best['Net_RRR_1D']:.2f}`",
        f"- Breakevens:    {be_strs}",
        f"- Est. Charges:  Rs `{best['Charges']:,.2f}`",
        "",
        "Portfolio Legs:"
    ])

    for leg in best['Portfolio']:
        action_str = "BUY " if leg['action'] == 'buy' else "SELL"
        price_lbl  = "Ask" if leg['action'] == 'buy' else "Bid"
        iv_pct     = leg.get('iv', 0) * 100
        lines.append(
            f"- {action_str} {leg['type']} {int(leg['strike'])} @ {price_lbl}: "
            f"Rs `{leg['entry_price']:.2f}` (IV: `{iv_pct:.1f}`%)"
        )

    lines.append("------")
    return "\n".join(lines)

# ==========================================
# LAMBDA HANDLER
# ==========================================
def lambda_handler(event, context):
    try:
        mongo_uri = get_secret('MONGO_URI') or os.getenv('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI not found.")

        client        = MongoClient(mongo_uri)
        spreads_col   = client['trading_data']['nifty_spreads']
        index_col     = client['trading_data']['nifty_5m_ohlcv']
        inference_col = client['deployments']['inference_results']

        latest_rec = spreads_col.find_one(sort=[("datetime", -1)])
        if not latest_rec:
            return {"statusCode": 200, "body": "Options DB empty."}

        latest_dt_str   = latest_rec.get('datetime')
        latest_date_str = latest_dt_str.split(' ')[0]

        # ── FIX 1: use 09:15 snapshot (market open) ──────────────────────────
        target_time_str = f"{latest_date_str} 09:15:00"

        inf_doc = inference_col.find_one({"Predicted_for_Date": latest_date_str})
        if not inf_doc:
            inf_doc = inference_col.find_one(sort=[("stored_at", -1)])
            if not inf_doc:
                return {"statusCode": 200, "body": "No inference data available."}

        pred_for  = inf_doc.get('Predicted_for_Date', 'Unknown')
        inference = {k: inf_doc[k] for k in [
            'Lower_85pct_CI_Price', 'Upper_85pct_CI_Price',
            'Lower_20pct_CI_Price', 'Upper_20pct_CI_Price',
            'Mean_Predicted_Price', 'Base_Price',
            'Prediction_Date', 'Predicted_for_Date',
            'Implied_Vol_Scaler',
        ] if k in inf_doc}

        df = pd.DataFrame(list(spreads_col.find({"datetime": target_time_str})))
        if df.empty:
            return {"statusCode": 200, "body": f"No snapshot rows found for {target_time_str}."}

        for col in ['strike', 'bid', 'ask', 'volume', 'ltp']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df[(df['volume'] > 0) & (df['bid'] > 0) & (df['ask'] > 0)].copy()
        df = (df.sort_values('datetime')
                .groupby(['strike', 'type', 'expiry'])
                .tail(1)
                .reset_index(drop=True))

        latest_idx = index_col.find_one({"datetime": target_time_str})
        spot_price = float(latest_idx['close']) if latest_idx else float(df['strike'].median())

        df['Date']   = pd.to_datetime(df['datetime']).dt.normalize()
        df           = df[df['type'].isin(['CE', 'PE'])].copy()
        df['Expiry'] = df['expiry'].apply(parse_fyers_expiry)
        df.dropna(subset=['Expiry'], inplace=True)
        df.rename(columns={'type': 'Type', 'strike': 'Strike', 'volume': 'Volume'}, inplace=True)

        buying_date        = df['Date'].iloc[0].date()
        max_exp_date       = buying_date + timedelta(days=MAX_EXPIRY_DAYS)

        available_expiries = sorted([
            d for d in df['Expiry'].unique()
            if pd.notna(d) and buying_date <= d.date() <= max_exp_date
        ])[:N_EXPIRIES_TO_CHECK]

        if not available_expiries:
            return {"statusCode": 200, "body": "No valid expiries in data."}

        is_bullish = inference.get('Mean_Predicted_Price', 0) >= inference.get('Base_Price', 0)

        # ── FIX 2: directional targets ────────────────────────────────────────
        # For BULLISH: target upper CI levels so bull call spread profits as
        #              market rises to/past those levels.
        # For BEARISH: target lower CI levels so bear put spread profits as
        #              market falls to/past those levels.
        # Dir1 = outer (85%) CI — larger move target
        # Dir2 = inner (20%) CI — conservative / higher-probability target
        target_dir1 = inference['Upper_85pct_CI_Price'] if is_bullish else inference['Lower_85pct_CI_Price']
        target_dir2 = inference['Upper_20pct_CI_Price'] if is_bullish else inference['Lower_20pct_CI_Price']

        # ── FIX 3: use scan_directional_spread (debit vertical spread aimed at
        #           the CI target) instead of the old get_vertical_spreads which
        #           mixed credit and debit spreads and pointed at the wrong CI
        #           bound, producing identical results and inverted breakevens. ─
        ic_results,   ic_tested   = scan_iron_condor(df, spot_price, inference, available_expiries, buying_date)
        ls_results,   ls_tested   = scan_long_strangle(df, spot_price, inference, available_expiries, buying_date)
        dir1_results, dir1_tested = scan_directional_spread(df, spot_price, is_bullish, target_dir1, available_expiries, buying_date)
        dir2_results, dir2_tested = scan_directional_spread(df, spot_price, is_bullish, target_dir2, available_expiries, buying_date)

        direction_str  = "Bullish" if is_bullish else "Bearish"
        spread_type    = "Bull Call Spread" if is_bullish else "Bear Put Spread"

        header = "\n".join([
            "**ML-GUIDED OPTIONS SCAN**",
            f"Time: `{target_time_str}`",
            f"Spot: Rs `{spot_price:,.2f}` | Pred for: `{pred_for}`",
            f"Signal: {direction_str} (Mean: Rs `{inference.get('Mean_Predicted_Price', 0):,.2f}`)",
            f"85% CI: Rs `{inference.get('Lower_85pct_CI_Price', 0):,.2f}` to Rs `{inference.get('Upper_85pct_CI_Price', 0):,.2f}`",
            f"20% CI: Rs `{inference.get('Lower_20pct_CI_Price', 0):,.2f}` to Rs `{inference.get('Upper_20pct_CI_Price', 0):,.2f}`",
            f"Results: IC=`{len(ic_results)}` | LS=`{len(ls_results)}` | Dir1=`{len(dir1_results)}` | Dir2=`{len(dir2_results)}`",
            "------"
        ])

        dir1_label = (
            f"Directional 1 — {spread_type} targeting {'Upper' if is_bullish else 'Lower'} 85% CI "
            f"(Rs `{target_dir1:,.2f}`)"
        )
        dir2_label = (
            f"Directional 2 — {spread_type} targeting {'Upper' if is_bullish else 'Lower'} 20% CI "
            f"(Rs `{target_dir2:,.2f}`)"
        )

        blocks = [
            _format_strategy_block("Iron Condor (Sell-Vol @ Outer 85% CI)",   "Iron Condor (85% CI)",   ic_results,   ic_tested),
            _format_strategy_block("Long Strangle (Buy-Vol @ Inner 20% CI)",  "Long Strangle (20% CI)", ls_results,   ls_tested),
            _format_strategy_block(dir1_label, spread_type, dir1_results, dir1_tested),
            _format_strategy_block(dir2_label, spread_type, dir2_results, dir2_tested),
        ]

        final_msg = header + "\n" + "\n".join(blocks)

        if len(final_msg) > 4000:
            final_msg = final_msg[:3990] + "\n`...(truncated)`"

        print(final_msg)
        send_telegram_alert(final_msg)
        return {"statusCode": 200, "body": "Scan complete, alert sent."}

    except Exception as e:
        error_trace = traceback.format_exc()
        error_type  = type(e).__name__
        msg         = f"{error_type}: {str(e)}"
        print("CRITICAL ERROR:\n", error_trace)
        alert_msg = f"**Algo Scanner Error:**\n{msg}\n\n**Traceback:**\n`{error_trace[-300:]}`"
        send_telegram_alert(alert_msg)
        return {"statusCode": 500, "body": f"Internal Error: {msg}"}