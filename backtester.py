import os
import argparse
import gc
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate

# ======================================================
# FLAGS
# ======================================================
REWRITE_RESULTS = True 
EXECUTE_ON_NEXT_OPEN = True
INCLUDE_TAXES = True

# ======================================================
# CONFIG & ENV
# ======================================================
load_dotenv()

DATA_FILE = os.getenv("DATA_FILE", "./data/nifty50_5m_20240101_20251231.csv")
SIGNAL_DIR = os.getenv("SIGNAL_DIR", "./signals")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
IND_RESULTS_DIR = os.getenv("IND_RESULTS_DIR", "./ind_results")
DEFAULT_STARTING_CASH = float(os.getenv("STARTING_CASH", 1_000_000.00))
BENCHMARK_SYMBOL = os.getenv("BENCHMARK_SYMBOL", "NSE:NIFTY50-INDEX")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IND_RESULTS_DIR, exist_ok=True)

RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252
SLIPPAGE = 0.0

MARKET_DATA_TIMEZONE_OFFSET = timedelta(hours=5, minutes=30) 

# ======================================================
# INDIAN TAX CONFIG (FYERS / OCT 2024 RATES)
# ======================================================
TAX_CONFIG = {
    "brokerage_rate": 0.0003,      
    "brokerage_cap": 20.0,         
    "stt_intraday_sell": 0.00025,  
    "stt_delivery": 0.001,         
    "txn_nse": 0.0000297,          
    "sebi_charges": 0.000001,      
    "stamp_duty_intra": 0.00003,   
    "stamp_duty_deliv": 0.00015,   
    "gst": 0.18                    
}

# ======================================================
# METRIC FUNCTIONS
# ======================================================
def calculate_sharpe(returns, annualization_factor):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    period_rf = RISK_FREE_RATE / annualization_factor
    excess = returns - period_rf
    return np.sqrt(annualization_factor) * excess.mean() / excess.std()

def calculate_sortino(returns, annualization_factor):
    if len(returns) < 2:
        return 0.0
    period_rf = RISK_FREE_RATE / annualization_factor
    excess = returns - period_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(annualization_factor) * excess.mean() / downside.std()

def calculate_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown

def calculate_taxes(entry_val, exit_val, qty, is_intraday=True):
    if not INCLUDE_TAXES: return 0.0
    
    buy_brok = min(entry_val * TAX_CONFIG["brokerage_rate"], TAX_CONFIG["brokerage_cap"])
    sell_brok = min(exit_val * TAX_CONFIG["brokerage_rate"], TAX_CONFIG["brokerage_cap"])
    total_brok = buy_brok + sell_brok

    txn_chg = (entry_val + exit_val) * TAX_CONFIG["txn_nse"]
    sebi_fee = (entry_val + exit_val) * TAX_CONFIG["sebi_charges"]
    gst_amt = (total_brok + txn_chg + sebi_fee) * TAX_CONFIG["gst"]

    if is_intraday:
        stt = exit_val * TAX_CONFIG["stt_intraday_sell"] 
    else:
        stt = (entry_val + exit_val) * TAX_CONFIG["stt_delivery"] 

    rate_stamp = TAX_CONFIG["stamp_duty_intra"] if is_intraday else TAX_CONFIG["stamp_duty_deliv"]
    stamp_duty = entry_val * rate_stamp

    return total_brok + txn_chg + sebi_fee + gst_amt + stt + stamp_duty

def format_duration(dt):
    if not isinstance(dt, timedelta):
        return "0m"
    days = dt.days
    total_seconds = dt.seconds
    minutes = (total_seconds % 3600) // 60
    hours = total_seconds // 3600
    display_min = (hours * 60) + minutes
    
    if days > 0:
        return f"{days}d {display_min}m"
    else:
        total_min_only = int(dt.total_seconds() / 60)
        return f"{total_min_only}m"

# ======================================================
# CORE BACKTESTER (AGGREGATE)
# ======================================================
def run_backtest(signal_file, starting_cash=None, cli_mode=False, plot_on_import=True):
    
    if starting_cash is None:
        initial_cash = DEFAULT_STARTING_CASH
    else:
        initial_cash = float(starting_cash)

    signal_path = os.path.join(SIGNAL_DIR, signal_file)
    if not os.path.exists(signal_path):
        raise FileNotFoundError(f"Signal file not found: {signal_path}")

    result_filename = os.path.basename(signal_file).replace(".csv", "_results.txt")
    result_path = os.path.join(RESULTS_DIR, result_filename)

    if os.path.exists(result_path) and not REWRITE_RESULTS:
        print(f"Skipping {signal_file}: Results exist and REWRITE_RESULTS=False.")
        return {}

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Market data file not found: {DATA_FILE}")

    raw_df = pd.read_csv(DATA_FILE)
    if not {"Symbol", "Epoch", "Close", "Open", "High", "Low"}.issubset(raw_df.columns):
        raise ValueError("Input file must contain Symbol, Epoch, Open, High, Low, Close columns.")

    raw_df["Datetime"] = pd.to_datetime(raw_df["Epoch"], unit="s") + MARKET_DATA_TIMEZONE_OFFSET

    market_pivot = raw_df.pivot_table(index="Datetime", columns="Symbol", values="Close").ffill().bfill()
    market_pivot_open = raw_df.pivot_table(index="Datetime", columns="Symbol", values="Open").ffill().bfill()
    market_pivot_high = raw_df.pivot_table(index="Datetime", columns="Symbol", values="High").ffill().bfill()
    market_pivot_low = raw_df.pivot_table(index="Datetime", columns="Symbol", values="Low").ffill().bfill()
    
    if market_pivot.empty: raise ValueError("Market data pivot is empty.")

    time_diffs = market_pivot.index.to_series().diff().dropna().head(100)
    median_diff = time_diffs.median()
    seconds = median_diff.total_seconds()
    
    if seconds >= 86400: 
        annualization_factor = TRADING_DAYS_PER_YEAR
        timeframe_label = "Days"
        is_intraday = False
    else:
        TRADING_SECONDS_PER_DAY = 22500 
        periods_per_day = TRADING_SECONDS_PER_DAY / seconds
        annualization_factor = int(TRADING_DAYS_PER_YEAR * periods_per_day)
        timeframe_label = "Minutes"
        is_intraday = True
    
    print(f"Detected Timeframe: {timeframe_label} ({int(seconds//60)}m)")

    signals = pd.read_csv(signal_path)
    if "Datetime" in signals.columns:
        signals["Datetime"] = pd.to_datetime(signals["Datetime"])
    elif "Epoch" in signals.columns:
        signals["Datetime"] = pd.to_datetime(signals["Epoch"], unit="s") + MARKET_DATA_TIMEZONE_OFFSET
    else:
        raise KeyError("Signal file is missing a 'Datetime' column.")

    signals = signals.sort_values(by="Datetime").reset_index(drop=True)
    if "TP" not in signals.columns: signals["TP"] = np.nan
    if "SL" not in signals.columns: signals["SL"] = np.nan
    
    sig_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    signals["Signal_Code"] = signals["Signal_text"].str.upper().map(sig_map).fillna(0)
    
    valid_mask = signals["Datetime"].isin(market_pivot.index)
    dropped_signals_df = signals[~valid_mask].copy()
    signals = signals[valid_mask]
    
    if not dropped_signals_df.empty:
        print(f"\nWARNING: {len(dropped_signals_df)} signals were dropped because timestamps are missing in Market Data.")

    # 5. Simulation State
    cash = initial_cash
    gross_cash = initial_cash # [NEW] Track Gross Cash (No Tax)
    
    positions = {} 
    trades = []
    rejected = []
    equity_curve = []       # Net Equity
    gross_equity_curve = [] # Gross Equity
    
    dates = market_pivot.index.tolist()
    signal_dict = {t: g for t, g in signals.groupby("Datetime")}

    # 6. Trading Loop
    tp_hits = 0       
    sl_hits = 0       
    signal_exits = 0  
    total_tax_paid = 0.0

    for i, current_time in enumerate(tqdm(dates, desc="Backtesting", unit="bar")):
        
        # [NEW] Bankruptcy Guard: Stop simulation if we are busted
        if cash <= 0:
            break

        # --- PESSIMISTIC TP/SL CHECK ---
        for sym in list(positions.keys()):
            pos = positions[sym]
            if sym not in market_pivot.columns: continue
            
            curr_close = market_pivot.at[current_time, sym]
            curr_high = market_pivot_high.at[current_time, sym] 
            curr_low = market_pivot_low.at[current_time, sym]   
            
            if pd.isna(curr_close): continue
            
            exit_type = None
            exec_price = curr_close
            
            # Check Long TP/SL
            if pos['qty'] > 0:
                if pd.notna(pos['sl']) and curr_low <= pos['sl']: 
                    exit_type = "SL"
                    exec_price = pos['sl'] 
                elif pd.notna(pos['tp']) and curr_high >= pos['tp']: 
                    exit_type = "TP"
                    exec_price = pos['tp']
            
            # Check Short TP/SL
            elif pos['qty'] < 0:
                if pd.notna(pos['sl']) and curr_high >= pos['sl']: 
                    exit_type = "SL"
                    exec_price = pos['sl']
                elif pd.notna(pos['tp']) and curr_low <= pos['tp']: 
                    exit_type = "TP"
                    exec_price = pos['tp']
            
            if exit_type:
                qty_abs = abs(pos['qty'])
                
                if pos['qty'] > 0: 
                    final_exit_price = exec_price * (1 - SLIPPAGE)
                    gross_pnl = (final_exit_price - pos['avg_price']) * qty_abs
                    tax = calculate_taxes(pos['avg_price']*qty_abs, final_exit_price*qty_abs, qty_abs, is_intraday)
                    
                    # [FIX] Subtract Tax from Cash!
                    cash += (final_exit_price * qty_abs) - tax
                    gross_cash += (final_exit_price * qty_abs)
                else: 
                    final_exit_price = exec_price * (1 + SLIPPAGE)
                    gross_pnl = (pos['avg_price'] - final_exit_price) * qty_abs
                    tax = calculate_taxes(pos['avg_price']*qty_abs, final_exit_price*qty_abs, qty_abs, is_intraday)
                    
                    cash += (pos['avg_price'] * qty_abs) + gross_pnl - tax 
                    gross_cash += (pos['avg_price'] * qty_abs) + gross_pnl
                
                total_tax_paid += tax
                net_pnl = gross_pnl - tax

                trades.append({
                    'Symbol': sym, 'Side': 'LONG' if pos['qty'] > 0 else 'SHORT',
                    'EntryPrice': pos['avg_price'], 'ExitPrice': final_exit_price,
                    'Qty': qty_abs, 'EntryTime': pos['entry_time'],
                    'ExitTime': current_time, 
                    'GrossPnL': gross_pnl, 'NetPnL': net_pnl, 'Tax': tax,
                    'ExitReason': exit_type
                })
                
                if exit_type == "TP": tp_hits += 1
                else: sl_hits += 1
                del positions[sym]
                continue 

        # --- PROCESS SIGNALS ---
        if current_time in signal_dict:
            group = signal_dict[current_time]
            buys = group[group["Signal_Code"] == 1]
            sells = group[group["Signal_Code"] == -1]

            def get_exec_price(sym, curr_idx):
                base_price = market_pivot.at[dates[curr_idx], sym]
                if EXECUTE_ON_NEXT_OPEN:
                    if curr_idx + 1 < len(dates):
                        next_t = dates[curr_idx + 1]
                        if sym in market_pivot_open.columns:
                            return market_pivot_open.at[next_t, sym]
                    else:
                        return None 
                return base_price

            # --- PROCESS BUY ---
            for _, row in buys.iterrows():
                symbol = row["Symbol"]
                qty_signal = int(row["Quantity"])
                row_tp = row["TP"] if pd.notna(row["TP"]) else None 
                row_sl = row["SL"] if pd.notna(row["SL"]) else None 

                if symbol not in market_pivot.columns: continue
                raw_price = get_exec_price(symbol, i)
                if raw_price is None: continue 

                entry_price_with_slip = raw_price * (1 + SLIPPAGE)
                pos = positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'entry_time': None})
                
                # Close Short
                if pos['qty'] < 0:
                    cover_qty = min(abs(pos['qty']), qty_signal)
                    exit_price_with_slip = raw_price * (1 + SLIPPAGE)
                    gross_pnl = (pos['avg_price'] - exit_price_with_slip) * cover_qty
                    
                    tax = calculate_taxes(pos['avg_price']*cover_qty, exit_price_with_slip*cover_qty, cover_qty, is_intraday)
                    total_tax_paid += tax
                    net_pnl = gross_pnl - tax
                    
                    cash += (pos['avg_price'] * cover_qty) + gross_pnl - tax
                    gross_cash += (pos['avg_price'] * cover_qty) + gross_pnl
                    
                    trades.append({
                        'Symbol': symbol, 'Side': 'SHORT', 'EntryPrice': pos['avg_price'],
                        'ExitPrice': exit_price_with_slip, 'Qty': cover_qty, 
                        'EntryTime': pos['entry_time'], 'ExitTime': current_time, 
                        'GrossPnL': gross_pnl, 'NetPnL': net_pnl, 'Tax': tax,
                        'ExitReason': 'Signal'
                    })
                    signal_exits += 1
                    pos['qty'] += cover_qty
                    qty_signal -= cover_qty
                    if pos['qty'] == 0: del positions[symbol]
                    else: positions[symbol] = pos

                # Open Long
                if qty_signal > 0:
                    # Basic Validation
                    if row_tp is not None and row_tp <= entry_price_with_slip:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': "Invalid Long TP"})
                        continue 
                    if row_sl is not None and row_sl >= entry_price_with_slip:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': "Invalid Long SL"})
                        continue 

                    cost = entry_price_with_slip * qty_signal
                    
                    # [NEW] Strict Cash Check (Prevent buying if broke)
                    if cash >= cost:
                        cash -= cost
                        gross_cash -= cost
                        if symbol in positions:
                            old = positions[symbol]
                            total_cost = (old['qty'] * old['avg_price']) + (qty_signal * entry_price_with_slip)
                            new_qty = old['qty'] + qty_signal
                            positions[symbol] = {
                                'qty': new_qty, 'avg_price': total_cost/new_qty, 
                                'entry_time': old['entry_time'], 'tp': row_tp, 'sl': row_sl
                            }
                        else:
                            positions[symbol] = {
                                'qty': qty_signal, 'avg_price': entry_price_with_slip, 
                                'entry_time': current_time, 'tp': row_tp, 'sl': row_sl
                            }
                    else:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': 'Insufficient Cash (Buy)'})

            # --- PROCESS SELL ---
            for _, row in sells.iterrows():
                symbol = row["Symbol"]
                qty_signal = int(row["Quantity"])
                row_tp = row["TP"] if pd.notna(row["TP"]) else None
                row_sl = row["SL"] if pd.notna(row["SL"]) else None

                if symbol not in market_pivot.columns: continue
                raw_price = get_exec_price(symbol, i)
                if raw_price is None: continue 

                entry_price_with_slip = raw_price * (1 - SLIPPAGE)
                pos = positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'entry_time': None})
                
                # Close Long
                if pos['qty'] > 0:
                    sell_qty = min(pos['qty'], qty_signal)
                    exit_price_with_slip = raw_price * (1 - SLIPPAGE)
                    gross_pnl = (exit_price_with_slip - pos['avg_price']) * sell_qty
                    
                    tax = calculate_taxes(pos['avg_price']*sell_qty, exit_price_with_slip*sell_qty, sell_qty, is_intraday)
                    total_tax_paid += tax
                    net_pnl = gross_pnl - tax

                    cash += (exit_price_with_slip * sell_qty) - tax 
                    gross_cash += (exit_price_with_slip * sell_qty)
                    
                    trades.append({
                        'Symbol': symbol, 'Side': 'LONG', 'EntryPrice': pos['avg_price'],
                        'ExitPrice': exit_price_with_slip, 'Qty': sell_qty, 
                        'EntryTime': pos['entry_time'], 'ExitTime': current_time, 
                        'GrossPnL': gross_pnl, 'NetPnL': net_pnl, 'Tax': tax,
                        'ExitReason': 'Signal'
                    })
                    signal_exits += 1
                    pos['qty'] -= sell_qty
                    qty_signal -= sell_qty
                    if pos['qty'] == 0: del positions[symbol]
                    else: positions[symbol] = pos
                
                # Open Short
                if qty_signal > 0:
                    if row_tp is not None and row_tp >= entry_price_with_slip:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': "Invalid Short TP"})
                        continue 
                    if row_sl is not None and row_sl <= entry_price_with_slip:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': "Invalid Short SL"})
                        continue 

                    cost = entry_price_with_slip * qty_signal
                    if cash >= cost:
                        cash -= cost 
                        gross_cash -= cost
                        if symbol in positions:
                            old = positions[symbol]
                            total_val = (abs(old['qty']) * old['avg_price']) + (qty_signal * entry_price_with_slip)
                            new_qty = abs(old['qty']) + qty_signal
                            positions[symbol] = {
                                'qty': -new_qty, 'avg_price': total_val/new_qty, 
                                'entry_time': old['entry_time'], 'tp': row_tp, 'sl': row_sl
                            }
                        else:
                            positions[symbol] = {
                                'qty': -qty_signal, 'avg_price': entry_price_with_slip, 
                                'entry_time': current_time, 'tp': row_tp, 'sl': row_sl
                            }
                    else:
                        rejected.append({'Time': current_time, 'Symbol': symbol, 'Qty': qty_signal, 'Reason': 'Insufficient Cash (Short)'})
        # M2M 
        m2m_total = 0.0
        for sym, pos in positions.items():
            if sym in market_pivot.columns:
                curr_price = market_pivot.at[current_time, sym]
                if pd.isna(curr_price): continue
                qty = pos['qty']
                if qty > 0: m2m_total += qty * curr_price
                else:
                    collateral = pos['avg_price'] * abs(qty)
                    unrealized_pnl = (pos['avg_price'] - curr_price) * abs(qty)
                    m2m_total += (collateral + unrealized_pnl)
        
        equity_curve.append(cash + m2m_total)
        gross_equity_curve.append(gross_cash + m2m_total)

    # ======================================================
    # METRICS & REPORTING
    # ======================================================
    equity_series = pd.Series(equity_curve, index=dates)
    gross_equity_series = pd.Series(gross_equity_curve, index=dates)
    
    returns_series = equity_series.pct_change().dropna()
    gross_returns_series = gross_equity_series.pct_change().dropna()
    
    final_equity = equity_series.iloc[-1]
    net_pnl_total = final_equity - initial_cash
    net_return = (net_pnl_total / initial_cash) * 100
    
    # Calculate Unrealized PnL at the end
    unrealized_pnl = 0.0
    for sym, pos in positions.items():
        curr = market_pivot.at[dates[-1], sym]
        qty = pos['qty']
        if qty > 0: unrealized_pnl += (curr - pos['avg_price']) * qty
        else: unrealized_pnl += (pos['avg_price'] - curr) * abs(qty)

    sum_gross_pnl = sum(t['GrossPnL'] for t in trades)
    sum_net_pnl = sum(t['NetPnL'] for t in trades)

    bench_return = 0.0
    bench_sharpe = 0.0
    benchmark_equity = None
    if BENCHMARK_SYMBOL in market_pivot.columns:
        bench_prices = market_pivot[BENCHMARK_SYMBOL]
        if not bench_prices.empty:
            benchmark_equity = (bench_prices / bench_prices.iloc[0]) * initial_cash 
            b_rets = bench_prices.pct_change().dropna()
            bench_sharpe = calculate_sharpe(b_rets, annualization_factor)
            bench_return = ((benchmark_equity.iloc[-1] - initial_cash) / initial_cash) * 100

    start_date, end_date = market_pivot.index[0], market_pivot.index[-1]
    duration_days = (end_date - start_date).days
    annualized_ret_pct = 0.0
    if duration_days > 0:
        years = duration_days / 365.25
        annualized_return = ((final_equity / initial_cash) ** (1 / years)) - 1
        annualized_ret_pct = annualized_return * 100
    
    dd_series = calculate_drawdown(equity_series)
    max_dd = dd_series.min() * 100
    avg_dd = dd_series.mean() * 100
    
    net_sharpe_val = calculate_sharpe(returns_series, annualization_factor)
    gross_sharpe_val = calculate_sharpe(gross_returns_series, annualization_factor)
    sortino_val = calculate_sortino(returns_series, annualization_factor)
    
    win_trades = [t for t in trades if t['NetPnL'] > 0]
    win_rate = (len(win_trades)/len(trades) * 100) if trades else 0.0
    largest_win = max([t['NetPnL'] for t in trades]) if trades else 0.0
    largest_loss = min([t['NetPnL'] for t in trades]) if trades else 0.0
    
    long_trades = len([t for t in trades if t['Side'] == 'LONG'])
    short_trades = len([t for t in trades if t['Side'] == 'SHORT'])

    durations = [] 
    for t in trades:
        dt = t['ExitTime'] - t['EntryTime']
        durations.append(dt)
        
    avg_hold_str = "0m"
    max_hold_str = "0m"
    
    if durations:
        avg_dt = pd.Series(durations).mean()
        max_dt = pd.Series(durations).max()
        avg_hold_str = format_duration(avg_dt)
        max_hold_str = format_duration(max_dt)

    with open(result_path, "w") as f:
        f.write(f"BACKTEST REPORT: {signal_file}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*85 + "\n\n") 
        
        # 1. Define the Full List of Metrics
        full_metrics = [
            ["Net Return (%)", f"{net_return:.2f}"],
            ["Annualized Return (%)", f"{annualized_ret_pct:.2f}"],
            ["Total Gross PnL (Realized)", f"{sum_gross_pnl:.2f}"], 
            ["Total Taxes Paid", f"{total_tax_paid:.2f}"], 
            ["Total Net PnL (Realized)", f"{sum_net_pnl:.2f}"], 
            ["Unrealized PnL (Open)", f"{unrealized_pnl:.2f}"],
            ["Benchmark Return (%)", f"{bench_return:.2f}"],
            ["Net Sharpe Ratio", f"{net_sharpe_val:.2f}"],
            ["Gross Sharpe Ratio", f"{gross_sharpe_val:.2f}"],
            ["Benchmark Sharpe", f"{bench_sharpe:.2f}"],
            ["Sortino Ratio", f"{sortino_val:.2f}"],
            ["Max Drawdown (%)", f"{max_dd:.2f}"],
            ["Avg Drawdown (%)", f"{avg_dd:.2f}"],
            ["Win Rate (%)", f"{win_rate:.2f}"],
            ["Largest Win", f"{largest_win:.2f}"],
            ["Largest Loss", f"{largest_loss:.2f}"],
            [f"Avg Holding", f"{avg_hold_str}"], 
            [f"Max Holding", f"{max_hold_str}"], 
            ["Final Cash", f"{cash:.2f}"],
            ["Total Trades", f"{len(trades)}"],
            ["Long Trades", f"{long_trades}"],
            ["Short Trades", f"{short_trades}"],
            ["Rejected Trades", f"{len(rejected)}"],
            ["Dropped Signals", f"{len(dropped_signals_df)}"], 
            ["TP Hits", f"{tp_hits}"],           
            ["SL Hits", f"{sl_hits}"],           
            ["Signal Exits", f"{signal_exits}"]
        ]
        
        # 2. Logic to Split into Double Columns
        mid_point = (len(full_metrics) + 1) // 2
        left_col = full_metrics[:mid_point]
        right_col = full_metrics[mid_point:]
        
        # Pad right column if odd number of items
        if len(right_col) < len(left_col):
            right_col.append(["", ""])
            
        double_col_data = []
        for l, r in zip(left_col, right_col):
            # [CHANGE] Insert "||" as a middle separator column
            double_col_data.append(l + [""] + r) 

        # 3. Write Table
        # [CHANGE] Added empty string "" in headers for the separator column
        f.write(tabulate(double_col_data, headers=["Metric", "Value", "", "Metric", "Value"], tablefmt="grid"))
        
        # ... (Rest of the file writing code remains unchanged)
        
        f.write(f"\n\nOPEN POSITIONS  ({len(positions)})\n")
        f.write("-" * 20 + "\n")
        if not positions: f.write("None\n")
        else:
            for sym, pos in positions.items():
                curr = market_pivot.at[dates[-1], sym]
                qty = pos['qty']
                if qty > 0: open_pnl = (curr - pos['avg_price']) * qty
                else: open_pnl = (pos['avg_price'] - curr) * abs(qty)
                f.write(f"Symbol: {sym}, Qty: {qty}, Entry: {pos['avg_price']:.2f}, OpenPnL: {open_pnl:.2f}\n")

        f.write(f"\nEXECUTED TRADES  ({len(trades)})\n")
        f.write("-" * 20 + "\n")
        if not trades: f.write("None\n")
        else:
            trade_rows = []
            for t in trades:
                dt = t['ExitTime'] - t['EntryTime']
                dur_str = format_duration(dt)
                
                trade_rows.append([
                    t['Symbol'], t['Side'], t['Qty'],
                    f"{t['EntryPrice']:.2f}", f"{t['ExitPrice']:.2f}", 
                    f"{t['GrossPnL']:.2f}", f"{t['Tax']:.2f}", f"{t['NetPnL']:.2f}", 
                    dur_str, t.get('ExitReason', '')
                ])
            f.write(tabulate(trade_rows, headers=["Symbol", "Side", "Qty", "Entry", "Exit", "Gross", "Tax", "Net", "Duration", "Reason"], tablefmt="simple"))

        f.write(f"\n\nREJECTED TRADES  ({len(rejected)})\n")
        f.write("-" * 20 + "\n")
        if not rejected: f.write("None\n")
        else:
            for r in rejected:
                f.write(f"Time: {r['Time']}, Symbol: {r['Symbol']}, Qty: {r['Qty']}, Reason: {r['Reason']}\n")

        # Log Dropped Signals explicitly in File
        f.write(f"\n\nDROPPED SIGNALS (Timestamp Mismatch)  ({len(dropped_signals_df)})\n")
        f.write("-" * 20 + "\n")
        if dropped_signals_df.empty:
            f.write("None\n")
        else:
            dropped_rows = []
            for _, r in dropped_signals_df.iterrows():
                dropped_rows.append([str(r["Datetime"]), r["Symbol"], r["Signal_text"]])
            f.write(tabulate(dropped_rows, headers=["Time", "Symbol", "Signal"], tablefmt="simple"))

    metrics = {
        "Net Return (%)": net_return,
        "Annualized Return (%)": annualized_ret_pct,
        "Net PnL": sum_net_pnl,
        "Total Tax": total_tax_paid,
        "Final Cash": cash,
        "Benchmark Return (%)": bench_return,
        "Max Drawdown (%)": max_dd,
        "Win Rate (%)": win_rate,
        "Net Sharpe": net_sharpe_val,
        "Gross Sharpe": gross_sharpe_val
    }

    display_metrics = {k: f"{v:.2f}" for k, v in metrics.items()}
    print("\n=== BACKTEST SUMMARY ===")
    print(tabulate(display_metrics.items(), headers=["Metric", "Value"], tablefmt="grid"))

    if not cli_mode and plot_on_import:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(equity_series.index, equity_series.values, label="Net Strategy", color='blue')
        ax1.plot(gross_equity_series.index, gross_equity_series.values, label="Gross Strategy", color='green', alpha=0.5, linestyle='--')
        if benchmark_equity is not None:
            ax1.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark", color='orange', alpha=0.7)
        ax1.set_title("Equity Curve (Net vs Gross vs Benchmark)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(dd_series.index, dd_series.values, label="Net Drawdown", color='red')
        ax2.fill_between(dd_series.index, dd_series.values, 0, color='red', alpha=0.3)
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    gc.collect()
    return metrics

def run_individual_backtests(signal_file, starting_cash=None, target_ticker=None):
    # Setup Cash
    if starting_cash is None:
        initial_cash = DEFAULT_STARTING_CASH
    else:
        initial_cash = float(starting_cash)

    signal_path = os.path.join(SIGNAL_DIR, signal_file)
    if not os.path.exists(signal_path): return

    # 1. Load Full Data (Once) for Benchmarking
    if not os.path.exists(DATA_FILE): return
    raw_df_full = pd.read_csv(DATA_FILE)
    raw_df_full["Datetime"] = pd.to_datetime(raw_df_full["Epoch"], unit="s") + MARKET_DATA_TIMEZONE_OFFSET
    
    # Calculate Benchmark Metrics (Common)
    bench_return = 0.0
    bench_sharpe = 0.0
    
    bench_data = raw_df_full[raw_df_full["Symbol"] == BENCHMARK_SYMBOL].sort_values("Datetime").set_index("Datetime")
    if not bench_data.empty:
        b_prices = bench_data["Close"]
        b_rets = b_prices.pct_change().dropna()
        
        # Estimate timeframe
        time_diffs = b_prices.index.to_series().diff().dropna().head(100)
        seconds = time_diffs.median().total_seconds() if not time_diffs.empty else 0
        
        if seconds >= 86400: annualization_factor = TRADING_DAYS_PER_YEAR
        else: 
             TRADING_SECONDS_PER_DAY = 22500 
             annualization_factor = int(TRADING_DAYS_PER_YEAR * (TRADING_SECONDS_PER_DAY / max(1, seconds)))
        
        bench_sharpe = calculate_sharpe(b_rets, annualization_factor)
        bench_return = ((b_prices.iloc[-1] - b_prices.iloc[0]) / b_prices.iloc[0]) * 100

    # 2. Load Signals
    all_signals = pd.read_csv(signal_path)
    if "Datetime" in all_signals.columns:
        all_signals["Datetime"] = pd.to_datetime(all_signals["Datetime"])
    elif "Epoch" in all_signals.columns:
        all_signals["Datetime"] = pd.to_datetime(all_signals["Epoch"], unit="s") + MARKET_DATA_TIMEZONE_OFFSET
    
    # Map Signals
    sig_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    all_signals["Signal_Code"] = all_signals["Signal_text"].str.upper().map(sig_map).fillna(0)

    # --- TARGET TICKER FILTERING ---
    if target_ticker:
        if target_ticker not in all_signals["Symbol"].values:
            print(f"Error: Target ticker '{target_ticker}' not found in signals.")
            return
        unique_tickers = [target_ticker]
        iterator = unique_tickers # No tqdm for single item
        print(f"\nRunning Single Backtest for: {target_ticker} (Cash: {initial_cash})")
    else:
        unique_tickers = all_signals["Symbol"].unique()
        iterator = tqdm(unique_tickers, desc="Processing")
        print(f"\nRunning Individual Breakdown for {len(unique_tickers)} tickers...")
    
    results_rows = []
    
    # 3. Loop Tickers
    for ticker in iterator:
        
        # A. Filter Data for Ticker
        ticker_signals = all_signals[all_signals["Symbol"] == ticker].copy().sort_values("Datetime").reset_index(drop=True)
        ticker_market = raw_df_full[raw_df_full["Symbol"] == ticker].copy().sort_values("Datetime")
        
        if ticker_market.empty: continue
        
        # B. Construct Pivot-like Structures (Single Series)
        p_close = ticker_market.set_index("Datetime")["Close"]
        p_open = ticker_market.set_index("Datetime")["Open"]
        p_high = ticker_market.set_index("Datetime")["High"]
        p_low = ticker_market.set_index("Datetime")["Low"]
        
        dates = ticker_market["Datetime"].tolist()
        
        # Recalc Timeframe
        time_diffs = pd.Series(dates).diff().dropna().head(100)
        seconds = time_diffs.median().total_seconds() if not time_diffs.empty else 0
        if seconds >= 86400: 
            annualization_factor = TRADING_DAYS_PER_YEAR; is_intraday=False
        else: 
            annualization_factor = int(TRADING_DAYS_PER_YEAR * (22500 / max(1, seconds)))
            is_intraday=True
            
        # C. Simulation State
        cash = initial_cash
        gross_cash = initial_cash
        position = None # {qty, avg_price, entry_time, tp, sl}
        trades = []
        rejected = [] # [FIX] Track rejected trades locally per ticker
        equity_curve = []
        gross_equity_curve = []
        
        # Group signals by time
        signal_dict = {t: g for t, g in ticker_signals.groupby("Datetime")}
        
        # Counters
        tp_hits = 0; sl_hits = 0; signal_exits = 0; total_tax = 0.0
        
        # D. Time Loop
        for i, current_time in enumerate(dates):
            if cash <= 0: break # Bankruptcy Guard
            
            # --- 1. Check Positions (TP/SL) ---
            if position is not None:
                # Use .get to handle potential missing timestamps safely
                curr_close = p_close.get(current_time)
                curr_high = p_high.get(current_time)
                curr_low = p_low.get(current_time)
                
                if pd.notna(curr_close):
                    exit_type = None
                    exec_price = curr_close
                    
                    # Long Check
                    if position['qty'] > 0:
                        if pd.notna(position['sl']) and curr_low <= position['sl']: 
                            exit_type="SL"; exec_price=position['sl']
                        elif pd.notna(position['tp']) and curr_high >= position['tp']: 
                            exit_type="TP"; exec_price=position['tp']
                    # Short Check
                    elif position['qty'] < 0:
                        if pd.notna(position['sl']) and curr_high >= position['sl']: 
                            exit_type="SL"; exec_price=position['sl']
                        elif pd.notna(position['tp']) and curr_low <= position['tp']: 
                            exit_type="TP"; exec_price=position['tp']
                        
                    if exit_type:
                        qty_abs = abs(position['qty'])
                        if position['qty'] > 0:
                            f_exit = exec_price * (1-SLIPPAGE)
                            g_pnl = (f_exit - position['avg_price']) * qty_abs
                            tax = calculate_taxes(position['avg_price']*qty_abs, f_exit*qty_abs, qty_abs, is_intraday)
                            cash += (f_exit*qty_abs) - tax
                            gross_cash += (f_exit*qty_abs)
                        else:
                            f_exit = exec_price * (1+SLIPPAGE)
                            g_pnl = (position['avg_price'] - f_exit) * qty_abs
                            tax = calculate_taxes(position['avg_price']*qty_abs, f_exit*qty_abs, qty_abs, is_intraday)
                            cash += (position['avg_price']*qty_abs) + g_pnl - tax
                            gross_cash += (position['avg_price']*qty_abs) + g_pnl
                            
                        total_tax += tax
                        trades.append({
                            'Side': 'LONG' if position['qty'] > 0 else 'SHORT', 
                            'NetPnL': g_pnl-tax, 
                            'GrossPnL': g_pnl, 
                            'ExitTime': current_time, 
                            'EntryTime': position['entry_time'],
                            'Qty': qty_abs,
                            'EntryPrice': position['avg_price'],
                            'ExitPrice': f_exit,
                            'ExitReason': exit_type
                        })
                        if exit_type=="TP": tp_hits+=1
                        else: sl_hits+=1
                        position = None # Clear Position
                        continue # Trade closed, move to next logic

            # --- 2. Process Signals ---
            if current_time in signal_dict:
                row = signal_dict[current_time].iloc[0]
                sig_code = row["Signal_Code"]
                qty_sig = int(row["Quantity"])
                r_tp = row["TP"] if pd.notna(row["TP"]) else None
                r_sl = row["SL"] if pd.notna(row["SL"]) else None
                
                # Exec Price Logic
                raw_price = p_close.get(current_time)
                if EXECUTE_ON_NEXT_OPEN and i+1 < len(dates):
                     next_t = dates[i+1]
                     raw_price = p_open.get(next_t)
                
                if pd.notna(raw_price):
                    
                    # BUY SIGNAL
                    if sig_code == 1:
                        entry = raw_price * (1+SLIPPAGE)
                        
                        # 1. Close Short if exists
                        if position and position['qty'] < 0:
                            c_qty = min(abs(position['qty']), qty_sig)
                            f_exit = raw_price * (1+SLIPPAGE)
                            g_pnl = (position['avg_price'] - f_exit) * c_qty
                            tax = calculate_taxes(position['avg_price']*c_qty, f_exit*c_qty, c_qty, is_intraday)
                            
                            cash += (position['avg_price']*c_qty) + g_pnl - tax
                            gross_cash += (position['avg_price']*c_qty) + g_pnl
                            total_tax += tax
                            
                            trades.append({
                                'Side': 'SHORT', 
                                'NetPnL': g_pnl-tax, 
                                'GrossPnL': g_pnl, 
                                'ExitTime': current_time, 
                                'EntryTime': position['entry_time'],
                                'Qty': c_qty,
                                'EntryPrice': position['avg_price'],
                                'ExitPrice': f_exit,
                                'ExitReason': 'Signal'
                            })
                            signal_exits += 1
                            
                            position['qty'] += c_qty
                            qty_sig -= c_qty # Remaining to buy
                            if position['qty'] == 0: position = None
                        
                        # 2. Open Long (if qty remaining)
                        if qty_sig > 0:
                            # VALIDATION
                            valid_signal = True
                            if r_tp is not None and r_tp <= entry:
                                rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Invalid Long TP"})
                                valid_signal = False
                            elif r_sl is not None and r_sl >= entry:
                                rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Invalid Long SL"})
                                valid_signal = False
                            
                            if valid_signal:
                                cost = entry * qty_sig
                                # Cash Check
                                if cash >= cost:
                                    cash -= cost
                                    gross_cash -= cost
                                    # Update/Create Position
                                    if position:
                                        t_cost = (position['qty']*position['avg_price']) + (qty_sig*entry)
                                        n_qty = position['qty'] + qty_sig
                                        position = {'qty': n_qty, 'avg_price': t_cost/n_qty, 'entry_time': position['entry_time'], 'tp': r_tp, 'sl': r_sl}
                                    else:
                                        position = {'qty': qty_sig, 'avg_price': entry, 'entry_time': current_time, 'tp': r_tp, 'sl': r_sl}
                                else:
                                    rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Insufficient Cash (Buy)"})

                    # SELL SIGNAL
                    elif sig_code == -1:
                        entry = raw_price * (1-SLIPPAGE)
                        
                        # 1. Close Long if exists
                        if position and position['qty'] > 0:
                            s_qty = min(position['qty'], qty_sig)
                            f_exit = raw_price * (1-SLIPPAGE)
                            g_pnl = (f_exit - position['avg_price']) * s_qty
                            tax = calculate_taxes(position['avg_price']*s_qty, f_exit*s_qty, s_qty, is_intraday)
                            
                            cash += (f_exit*s_qty) - tax 
                            gross_cash += (f_exit*s_qty)
                            total_tax += tax
                            
                            trades.append({
                                'Side': 'LONG', 
                                'NetPnL': g_pnl-tax, 
                                'GrossPnL': g_pnl, 
                                'ExitTime': current_time, 
                                'EntryTime': position['entry_time'],
                                'Qty': s_qty,
                                'EntryPrice': position['avg_price'],
                                'ExitPrice': f_exit,
                                'ExitReason': 'Signal'
                            })
                            signal_exits += 1
                            
                            position['qty'] -= s_qty
                            qty_sig -= s_qty
                            if position['qty'] == 0: position = None
                        
                        # 2. Open Short
                        if qty_sig > 0:
                            # VALIDATION
                            valid_signal = True
                            if r_tp is not None and r_tp >= entry:
                                rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Invalid Short TP"})
                                valid_signal = False
                            elif r_sl is not None and r_sl <= entry:
                                rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Invalid Short SL"})
                                valid_signal = False
                                
                            if valid_signal:
                                cost = entry * qty_sig
                                if cash >= cost:
                                    cash -= cost 
                                    gross_cash -= cost
                                    if position:
                                        t_val = (abs(position['qty'])*position['avg_price']) + (qty_sig*entry)
                                        n_qty = abs(position['qty']) + qty_sig
                                        position = {'qty': -n_qty, 'avg_price': t_val/n_qty, 'entry_time': position['entry_time'], 'tp': r_tp, 'sl': r_sl}
                                    else:
                                        position = {'qty': -qty_sig, 'avg_price': entry, 'entry_time': current_time, 'tp': r_tp, 'sl': r_sl}
                                else:
                                    rejected.append({'Time': current_time, 'Symbol': ticker, 'Qty': qty_sig, 'Reason': "Insufficient Cash (Short)"})

            # --- 3. Mark to Market ---
            m2m = 0.0
            if position:
                curr = p_close.get(current_time)
                if pd.notna(curr):
                    if position['qty'] > 0: m2m = position['qty'] * curr
                    else: m2m = (position['avg_price']*abs(position['qty'])) + ((position['avg_price'] - curr)*abs(position['qty']))
            
            equity_curve.append(cash + m2m)
            gross_equity_curve.append(gross_cash + m2m)
            
        # --- E. Calculate Metrics for Ticker ---
        actual_dates = dates[:len(equity_curve)]
        eq_s = pd.Series(equity_curve, index=actual_dates)
        g_eq_s = pd.Series(gross_equity_curve, index=actual_dates)
        
        if eq_s.empty: continue
        
        rets = eq_s.pct_change().dropna()
        g_rets = g_eq_s.pct_change().dropna()
        
        final_eq = eq_s.iloc[-1]
        net_ret = ((final_eq - initial_cash)/initial_cash)*100
        
        dur_days = (actual_dates[-1] - actual_dates[0]).days
        ann_ret = 0.0
        if dur_days > 0 and final_eq > 0:
            ann_ret = ((final_eq/initial_cash)**(1/(dur_days/365.25)) - 1) * 100
            
        sum_gross = sum(t['GrossPnL'] for t in trades)
        sum_net = sum(t['NetPnL'] for t in trades)
        
        unrealized = 0.0
        if position:
            curr = p_close.get(actual_dates[-1])
            if pd.notna(curr):
                if position['qty'] > 0: unrealized = (curr - position['avg_price']) * position['qty']
                else: unrealized = (position['avg_price'] - curr) * abs(position['qty'])
            
        n_sharpe = calculate_sharpe(rets, annualization_factor)
        g_sharpe = calculate_sharpe(g_rets, annualization_factor)
        sortino = calculate_sortino(rets, annualization_factor)
        
        dd = calculate_drawdown(eq_s)
        max_dd = dd.min() * 100
        avg_dd = dd.mean() * 100
        
        wins = [t for t in trades if t['NetPnL'] > 0]
        win_rate = (len(wins)/len(trades)*100) if trades else 0.0
        l_win = max([t['NetPnL'] for t in trades]) if trades else 0.0
        l_loss = min([t['NetPnL'] for t in trades]) if trades else 0.0
        
        durations = [(t['ExitTime']-t['EntryTime']).total_seconds()/60 for t in trades]
        avg_hold = np.mean(durations) if durations else 0
        max_hold = np.max(durations) if durations else 0
        
        long_trades_count = len([t for t in trades if t['Side'] == 'LONG'])
        short_trades_count = len([t for t in trades if t['Side'] == 'SHORT'])
        
        metric_dict = {
            "Ticker": ticker,
            "Testing Period": f"{actual_dates[0].date()} to {actual_dates[-1].date()}",
            "Net Return (%)": net_ret,
            "Annualised Return (%)": ann_ret,
            "Gross PnL (Realised)": sum_gross,
            "Total Taxes": total_tax,
            "Net PnL (Realised)": sum_net,
            "Unrealised PnL (Open)": unrealized,
            "Benchmark Return (%)": bench_return,
            "Net Sharpe": n_sharpe,
            "Gross Sharpe": g_sharpe,
            "Benchmark Sharpe": bench_sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": max_dd,
            "Avg Drawdown (%)": avg_dd,
            "Win Rate (%)": win_rate,
            "Largest Win": l_win,
            "Largest Loss": l_loss,
            "Avg Holding (mins)": avg_hold,
            "Max Holding (mins)": max_hold,
            "Final Cash": cash,
            "Total Trades": len(trades),
            "Long Trades": long_trades_count,
            "Short Trades": short_trades_count,
            "Open Position": 1 if position else 0,
            "Rejected Trades": len(rejected),
            "TP Hits": tp_hits,
            "SL Hits": sl_hits,
            "Strategy Exits": signal_exits
        }

        # --- SPECIAL HANDLING FOR TARGET TICKER ---
        if target_ticker and ticker == target_ticker:
            # 1. Output Equity & Drawdown Curves
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # --- NET EQUITY (With Taxes) ---
            ax1.plot(eq_s.index, eq_s.values, label=f"Net Equity (Strategy)", color='blue')
            
            # --- GROSS EQUITY (No Taxes) ---
            ax1.plot(g_eq_s.index, g_eq_s.values, label=f"Gross Equity (No Tax)", color='green', linestyle='--', alpha=0.6)
            
            # --- BENCHMARK EQUITY ---
            if not bench_data.empty:
                # Reindex benchmark to match our strategy dates
                aligned_bench = bench_data["Close"].reindex(actual_dates).ffill().bfill()
                # Normalize benchmark to start at initial_cash
                if not aligned_bench.empty and aligned_bench.iloc[0] > 0:
                     normalized_bench = (aligned_bench / aligned_bench.iloc[0]) * initial_cash
                     ax1.plot(normalized_bench.index, normalized_bench.values, label="Benchmark (Nifty)", color='orange', alpha=0.7)

            ax1.set_title(f"Equity Curve: {ticker}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(dd.index, dd.values, label="Drawdown", color='red')
            ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

            # 2. PRINT Report to Console (No File Save)
            print(f"SINGLE TICKER BACKTEST: {ticker}")
            print("="*60)
            
            # Format metrics for report
            formatted_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in metric_dict.items()]
            print(tabulate(formatted_data, headers=["Metric", "Value"], tablefmt="grid"))
            
            # 3. Print REJECTED Trades (Instead of executed)
            print(f"\nREJECTED TRADES ({len(rejected)})")
            print("-" * 20)
            if rejected:
                r_rows = []
                for r in rejected:
                    r_rows.append([r['Time'], r['Qty'], r['Reason']])
                print(tabulate(r_rows, headers=["Time", "Qty", "Reason"], tablefmt="simple"))
            else:
                print("None")
                
            return 

        # --- NORMAL AGGREGATION ---
        results_rows.append(metric_dict)
        
    # Save CSV (Only if running batch mode)
    if not target_ticker:
        out_name = os.path.basename(signal_file).replace(".csv", "_breakdown.csv")
        out_path = os.path.join(IND_RESULTS_DIR, out_name)
        pd.DataFrame(results_rows).to_csv(out_path, index=False)
        print(f"\nIndividual breakdown saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--cash", type=float, default=None, help="Override starting cash")
    args = parser.parse_args()
    
    run_backtest(args.signals, starting_cash=args.cash, cli_mode=True)
    run_individual_backtests(args.signals, starting_cash=args.cash)