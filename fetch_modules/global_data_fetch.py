import os
import glob
import json
import argparse
import tempfile
import subprocess
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

def get_timeframe_delta(tf_str: str) -> pd.Timedelta:
    """Converts a timeframe string like '5m' or '1h' to a pandas Timedelta."""
    if tf_str.endswith('m'):
        return pd.Timedelta(minutes=int(tf_str[:-1]))
    elif tf_str.endswith('h'):
        return pd.Timedelta(hours=int(tf_str[:-1]))
    elif tf_str.endswith('d'):
        return pd.Timedelta(days=int(tf_str[:-1]))
    return pd.Timedelta(minutes=5) # Default fallback


def download_market_data(start_date: str, end_date: str, timeframe: str = '5m'):
    load_dotenv()

    os.environ['NODE_OPTIONS'] = '--dns-result-order=ipv4first'

    data_dir = os.getenv("DATA_DIR", "./data")
    log_dir = os.getenv("LOG_DIR", "./logs")
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, 'dukascopy_fetch_errors.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    symbols = {
        "sp500": "usa500idxusd",
        "dollar_index": "dollaridxusd",
        "hang_seng": "hkgidxhkd",
        "crudeoil": "lightcmdusd"
    }

    tf_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15', '30m': 'm30', '1h': 'h1', '1d': 'd1'}
    duka_tf = tf_map.get(timeframe, timeframe)
    target_td = get_timeframe_delta(timeframe)
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"

    working_file = os.path.join(data_dir, f"working_data_temp_{timeframe}.csv")
    if os.path.exists(working_file):
        os.remove(working_file) # Clean up from any previous crashed runs

    # ---------------------------------------------------------
    # 1. Existing File Detection & Per-Symbol Date Analysis
    # ---------------------------------------------------------
    raw_files = glob.glob(os.path.join(data_dir, "global_data_*.csv"))
    existing_files = [f for f in raw_files if "working_data_temp" not in f]
    valid_existing_files = []
    
    for f in existing_files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            if len(df_head) > 1 and 'timestamp' in df_head.columns:
                ts = pd.to_datetime(df_head['timestamp'])
                diffs = ts.diff().dropna()
                
                if not diffs.empty:
                    min_diff = diffs[diffs > pd.Timedelta(0)].min()
                    if min_diff == target_td:
                        valid_existing_files.append(f)
                    else:
                        print(f"Skipping {os.path.basename(f)} - Data interval ({min_diff}) does not match requested ({timeframe}).")
        except Exception as e:
            logging.error(f"Error checking timeframe for {f}: {e}")
            print(f"Skipping {os.path.basename(f)} due to read error.")

    symbol_daily_counts = {sym: {} for sym in symbols.keys()}
    
    if valid_existing_files:
        print(f"Found existing {timeframe} dataset(s). Analyzing dates per symbol to detect partial data...")
        first_file_write = True
        for f in valid_existing_files:
            for chunk in pd.read_csv(f, chunksize=100000):
                # Extract date safely without modifying the original timestamp column
                chunk['date'] = pd.to_datetime(chunk['timestamp']).dt.date
                
                # Count the number of rows per day per symbol
                counts = chunk.groupby(['symbol', 'date']).size().reset_index(name='count')
                for _, row in counts.iterrows():
                    sym_name = row['symbol']
                    dt_val = row['date']
                    c = row['count']
                    if sym_name in symbol_daily_counts:
                        symbol_daily_counts[sym_name][dt_val] = symbol_daily_counts[sym_name].get(dt_val, 0) + c
                
                # Clean up and copy to working file
                chunk.drop(columns=['date'], inplace=True)
                chunk.to_csv(working_file, mode='a', index=False, header=first_file_write)
                first_file_write = False

    # Calculate exact missing or partial ranges specifically per symbol
    req_start_dt = pd.to_datetime(start_date).date()
    req_end_dt = pd.to_datetime(end_date).date()
    ranges_to_fetch_per_symbol = {}
    any_downloads_needed = False
    
    for name in symbols.keys():
        dates_to_fetch = []
        counts = symbol_daily_counts.get(name, {})
        max_candles = max(counts.values()) if counts else 0
        
        sym_min_date = min(counts.keys()) if counts else None
        sym_max_date = max(counts.keys()) if counts else None
        
        # DST Tolerance: Allow 20% natural variance to prevent standard days 
        # from being incorrectly flagged when compared to an elongated DST day.
        candle_threshold = max_candles * 0.80 
        
        curr_d = req_start_dt
        while curr_d <= req_end_dt:
            if curr_d.weekday() < 5: # only weekdays
                if curr_d not in counts:
                    # Smart Holiday Check: If the missing day is INSIDE our recorded data bounds, 
                    # it was almost certainly dropped by our flatline filter. We skip it to avoid fetch loops.
                    if sym_min_date and sym_max_date and (sym_min_date < curr_d < sym_max_date):
                        pass 
                    else:
                        dates_to_fetch.append(curr_d) # Genuinely missing data (outside bounds)
                elif curr_d == sym_max_date:
                    dates_to_fetch.append(curr_d) # Always fetch the absolute latest day to update live data
                elif counts[curr_d] < candle_threshold:
                    dates_to_fetch.append(curr_d) # Historical day with significant partial data missing
            curr_d += pd.Timedelta(days=1)
            
        if not dates_to_fetch:
            ranges_to_fetch_per_symbol[name] = []
            continue
            
        any_downloads_needed = True
        dates_to_fetch.sort()
        ranges = []
        start_range = dates_to_fetch[0]
        prev_date = dates_to_fetch[0]
        
        # Group missing dates into continuous blocks to minimize API overhead
        for d in dates_to_fetch[1:]:
            if (d - prev_date).days <= 5: # Small bridge to keep ranges together over weekends
                prev_date = d
            else:
                ranges.append((start_range, prev_date))
                start_range = d
                prev_date = d
        ranges.append((start_range, prev_date))
        ranges_to_fetch_per_symbol[name] = ranges

    if not any_downloads_needed:
        print(f"Requested range ({start_date} to {end_date}) is already fully complete for all symbols.")
        print("Proceeding to clean and verify existing data.")

    # ---------------------------------------------------------
    # 2. Download Missing/Partial Data with TQDM Progress
    # ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, sym in symbols.items():
            ranges = ranges_to_fetch_per_symbol.get(name, [])
            if not ranges:
                continue
                
            print(f"\n--- Fetching updates for {name} ({timeframe}) ---")
            total_days = sum((r[1] - r[0]).days + 1 for r in ranges)
            batch_buffer = []
            days_accumulated = 0
            
            with tqdm(total=total_days, desc=f"Fetching {name:<12}", unit="day") as pbar:
                for r_start, r_end in ranges:
                    curr_dt = r_start
                    while curr_dt <= r_end:
                        # Dynamic chunking: 90 days for high timeframes, 1 day for minute timeframes to avoid socket crashes
                        chunk_limit = 89 if timeframe in ['1h', '1d'] else 0
                        chunk_end = min(curr_dt + pd.Timedelta(days=chunk_limit), r_end)
                        
                        c_start_str = curr_dt.strftime('%Y-%m-%d')
                        # Push the CLI end date forward by 1 day because the '-to' argument is exclusive 
                        cli_end_dt = chunk_end + pd.Timedelta(days=1)
                        c_end_str = cli_end_dt.strftime('%Y-%m-%d')
                        days_in_chunk = (chunk_end - curr_dt).days + 1
                        
                        # Dynamic Backoff Variables
                        max_retries = 4
                        current_pause_ms = 0
                        max_pause_ms = 10000
                        fetch_success = False

                        for attempt in range(max_retries + 1):
                            # Included --yes to ensure npx doesn't crash on prompt
                            cmd = [
                                npx_cmd, "--yes", "dukascopy-node",
                                "-i", sym, "-from", c_start_str, "-to", c_end_str,
                                "-t", duka_tf, "-f", "csv", "-dir", temp_dir
                            ]
                            
                            # Only inject the batch pause argument if we've scaled it up
                            if current_pause_ms > 0:
                                cmd.extend(["-bp", str(current_pause_ms)])
                                
                            try:
                                subprocess.run(cmd, check=True, capture_output=True)
                                fetch_success = True
                                break  # Break loop on success
                            except subprocess.CalledProcessError as e:
                                err_str = e.stderr.decode('utf-8').strip() if e.stderr else ""
                                out_str = e.stdout.decode('utf-8').strip() if e.stdout else ""
                                full_err_msg = " | ".join(filter(None, [err_str, out_str])) or str(e)
                                
                                if attempt < max_retries:
                                    # Scale up the pause time (0 -> 1000 -> 2500 -> 6250 -> 10000)
                                    current_pause_ms = min(max_pause_ms, max(1000, int(current_pause_ms * 2.5) if current_pause_ms > 0 else 1000))
                                    logging.warning(f"Attempt {attempt + 1} failed for {name} ({c_start_str}). Retrying with {current_pause_ms}ms pause. Error: {full_err_msg}")
                                else:
                                    logging.error(f"Failed to fetch {name} for {c_start_str} to {c_end_str} after {max_retries} retries. Error: {full_err_msg}")
                                    tqdm.write(f"Logged API error for {name}. Check logs.")

                        if not fetch_success:
                            pbar.update(days_in_chunk)
                            curr_dt = chunk_end + pd.Timedelta(days=1)
                            continue
                            
                        csv_files = glob.glob(os.path.join(temp_dir, f"*{sym}*.csv"))
                        if not csv_files:
                            logging.error(f"Missing CSV generated for {name} for {c_start_str} to {c_end_str}")
                            pbar.update(days_in_chunk)
                            curr_dt = chunk_end + pd.Timedelta(days=1)
                            continue
                            
                        target_csv = csv_files[0]
                        
                        try:
                            df = pd.read_csv(target_csv)
                        except pd.errors.EmptyDataError:
                            os.remove(target_csv)
                            pbar.update(days_in_chunk)
                            curr_dt = chunk_end + pd.Timedelta(days=1)
                            continue

                        # Clean and format the valid data chunk
                        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
                        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
                        
                        if time_col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[time_col]):
                                dt_series = pd.to_datetime(df[time_col], unit='ms')
                            else:
                                dt_series = pd.to_datetime(df[time_col])
                            
                            mask = dt_series.dt.dayofweek < 5
                            df = df[mask].copy()
                            dt_series = dt_series[mask]
                            
                            df['timestamp'] = dt_series
                            if time_col != 'timestamp':
                                df.drop(columns=[time_col], inplace=True)
                                
                        df.insert(1, 'symbol', name)
                        batch_buffer.append(df)
                        os.remove(target_csv)
                        
                        days_accumulated += days_in_chunk
                        pbar.update(days_in_chunk)
                        
                        # --- 90 DAY BATCH FLUSH TO DISK ---
                        if days_accumulated >= 90:
                            if batch_buffer:
                                batch_df = pd.concat(batch_buffer, ignore_index=True)
                                batch_df.to_csv(working_file, mode='a', index=False, header=not os.path.exists(working_file))
                                batch_buffer.clear()
                            days_accumulated = 0
                            
                        curr_dt = chunk_end + pd.Timedelta(days=1)
                        
            # Flush any remaining data in the buffer after the symbol finishes
            if batch_buffer:
                batch_df = pd.concat(batch_buffer, ignore_index=True)
                batch_df.to_csv(working_file, mode='a', index=False, header=not os.path.exists(working_file))
                batch_buffer.clear()

    # ---------------------------------------------------------
    # 3. Aggregation, Deduplication & Export
    # ---------------------------------------------------------
    if not os.path.exists(working_file):
        print("\nNo valid data found to process.")
        return

    print(f"\nProcessing final {timeframe} dataset (Deduplicating, Cleaning & Sorting)...")
    
    # Load the entire prepared file into memory once at the end for exact sorting
    final_df = pd.read_csv(working_file)
    
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    
    # The keep='last' rule ensures that newly fetched, completed rows correctly overwrite the older partial rows
    final_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last', inplace=True)
    final_df.sort_values(by=['symbol', 'timestamp'], inplace=True)
    
    # --- HOLIDAY & FLATLINE FILTERING ---
    final_df['date'] = final_df['timestamp'].dt.date
    daily_stats = final_df.groupby(['symbol', 'date']).agg({'high': 'max', 'low': 'min'}).reset_index()
    flat_days = daily_stats[daily_stats['high'] == daily_stats['low']]
    num_removed_days = len(flat_days)
    
    if num_removed_days > 0:
        flat_days['sym_date'] = flat_days['symbol'] + "_" + flat_days['date'].astype(str)
        final_df['sym_date'] = final_df['symbol'] + "_" + final_df['date'].astype(str)
        
        flat_sym_dates = set(flat_days['sym_date'])
        mask = final_df['sym_date'].isin(flat_sym_dates)
        
        final_df = final_df[~mask].copy()
        print(f"Removed {num_removed_days} day(s) due to no movement.")
        final_df.drop(columns=['date', 'sym_date'], inplace=True)
    else:
        print(f"Removed 0 day(s) due to no movement. (Dataset is already clean)")
        final_df.drop(columns=['date'], inplace=True)

    # Failsafe in case cleaning emptied the dataframe
    if final_df.empty:
        print("Dataset is empty after cleaning flatlines. Aborting save.")
        os.remove(working_file)
        return
    # ------------------------------------

    final_df['timestamp'] = final_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    file_min_date = pd.to_datetime(final_df['timestamp']).min().strftime('%Y%m%d')
    file_max_date = pd.to_datetime(final_df['timestamp']).max().strftime('%Y%m%d')
    
    if pd.isna(file_min_date) or pd.isna(file_max_date):
        new_filename = f"global_data_{timeframe}_unknown_dates.csv"
    else:
        new_filename = f"global_data_{timeframe}_{file_min_date}_{file_max_date}.csv"
        
    new_filepath = os.path.join(data_dir, new_filename)

    final_df.to_csv(new_filepath, index=False)
    print(f"Success! Saved aggregated {timeframe} data to: {new_filepath}")

    # Cleanup working file and strictly ONLY old files that matched our timeframe
    os.remove(working_file)
    if valid_existing_files:
        for old_file in valid_existing_files:
            if os.path.abspath(old_file) != os.path.abspath(new_filepath):
                os.remove(old_file)
                print(f"Cleaned up old {timeframe} file: {old_file}")

    universe_path = os.path.join(data_dir, "global_universe.json")
    with open(universe_path, 'w') as f:
        json.dump(list(symbols.keys()), f, indent=4)
    print(f"Success! Saved universe map to: {universe_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dukascopy Smart Data Downloader Pipeline")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Resolution (e.g., 1m, 5m, 1h, 1d)")
    
    args = parser.parse_args()
    download_market_data(args.start, args.end, args.timeframe)