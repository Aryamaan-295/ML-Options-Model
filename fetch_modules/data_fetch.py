import csv
import time
import os
import json
import gc
import shutil
import tempfile
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from tqdm import tqdm
import pandas as pd

# ==========================================
# STRICT RATE LIMITER (Token Bucket)
# ==========================================
class StrictRateLimiter:
    """
    Enforces 10 req/sec AND 200 req/min limits strictly.
    Pauses execution if the minute bucket is empty.
    """
    def __init__(self):
        # Per Second: 10 reqs (Set to 9 for safety)
        self.sec_rate = 9
        self.sec_tokens = self.sec_rate
        self.last_sec_update = time.time()
        
        # Per Minute: 200 reqs (Set to 190 for safety)
        self.min_rate = 190
        self.min_tokens = self.min_rate
        self.last_min_update = time.time()

    def wait(self):
        while True:
            now = time.time()
            
            # Refill Second Tokens
            if now - self.last_sec_update >= 1.0:
                self.sec_tokens = self.sec_rate
                self.last_sec_update = now
                
            # Refill Minute Tokens
            if now - self.last_min_update >= 60.0:
                self.min_tokens = self.min_rate
                self.last_min_update = now
                
            # Check availability
            if self.sec_tokens > 0 and self.min_tokens > 0:
                self.sec_tokens -= 1
                self.min_tokens -= 1
                return # Allow request
            
            # If out of tokens, wait
            if self.min_tokens <= 0:
                wait_time = 60.0 - (now - self.last_min_update) + 1.0
                if wait_time > 0:
                    # Log only if wait is significant
                    if wait_time > 2:
                        print(f" [Rate Limit] Max reqs/min reached. Pausing for {wait_time:.1f}s...")
                    time.sleep(wait_time)
            else:
                time.sleep(0.1) # Short wait for second limit

limiter = StrictRateLimiter()

# ==========================================
# FILE HELPERS
# ==========================================
def timeframe_tag():
    return f"{TIMEFRAME}m" if TIMEFRAME.isdigit() else TIMEFRAME.lower()

def base_filename():
    return f"{UNIVERSE_TAG}_{timeframe_tag()}"

def full_filename(start, end):
    return f"{base_filename()}_{start.replace('-','')}_{end.replace('-','')}.csv"

def find_existing_file(existing_data_file):
    if existing_data_file and os.path.exists(existing_data_file):
        return existing_data_file
        
    for f in os.listdir(DATA_DIR):
        if f.startswith(base_filename()) and f.endswith(".csv"):
            return os.path.join(DATA_DIR, f)
    return None

def update_env_file(key, value):
    env_path = ".env" 
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    key_found = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}=\"{value}\"\n")
            key_found = True
        else:
            new_lines.append(line)
    
    if not key_found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}=\"{value}\"\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)

# ==========================================
# DATE UTILITIES
# ==========================================
def split_ranges(start, end):
    ranges = []
    s = start
    if s > end: return []
        
    while s <= end:
        e = min(s + timedelta(days=MAX_CHUNK_DAYS - 1), end)
        ranges.append((s, e))
        s = e + timedelta(days=1)
    return ranges

def scan_symbol_bounds(path):
    """
    Scans the file and builds a map of existing data ranges per symbol.
    Returns: { 'SBIN-EQ': [min_date, max_date], ... }
    """
    bounds = {}
    if not os.path.exists(path): return bounds
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sym = r["Symbol"]
            try:
                ep = float(r["Epoch"])
                dt = datetime.fromtimestamp(ep)
                
                if sym not in bounds:
                    bounds[sym] = [dt, dt]
                else:
                    if dt < bounds[sym][0]: bounds[sym][0] = dt
                    if dt > bounds[sym][1]: bounds[sym][1] = dt
            except (ValueError, KeyError):
                continue
    return bounds

# ==========================================
# FINAL SORT & DEDUPLICATE
# ==========================================
def scan_date_bounds(path):
    """Safely scans a file to find min/max dates WITHOUT modifying the file in any way."""
    print("No new data added. Scanning existing file to verify dates for renaming...")
    min_epoch = float('inf')
    max_epoch = float('-inf')
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    ep = int(float(r["Epoch"]))
                    if ep < min_epoch: min_epoch = ep
                    if ep > max_epoch: max_epoch = ep
                except (ValueError, KeyError):
                    continue
        if min_epoch != float('inf') and max_epoch != float('-inf'):
            return min_epoch, max_epoch
    except Exception as e:
        print(f"Error scanning bounds: {e}")
    return None, None

def sort_csv(path):
    print("[PROCESS]: Sorting and removing duplicates safely.")
    try:
        # Load directly into highly-optimized Pandas arrays, not Python dicts
        df = pd.read_csv(path)
        
        if df.empty:
            return None, None
            
        # Deduplicate and sort in place
        df.drop_duplicates(subset=['Symbol', 'Epoch'], keep='last', inplace=True)
        df.sort_values(by=['Symbol', 'Epoch'], inplace=True)
        
        min_epoch = int(df['Epoch'].min())
        max_epoch = int(df['Epoch'].max())

        # Overwrite safely
        tmp_name = path + ".tmp"
        df.to_csv(tmp_name, index=False)
        shutil.move(tmp_name, path)
        
        # Force garbage collection to free RAM immediately
        del df
        gc.collect()
        
        return min_epoch, max_epoch

    except Exception as e:
        print(f"[ERROR]: Error during sorting: {e}")
        return None, None
    
# ==========================================
# FYERS FETCH
# ==========================================
def fetch_history(symbol, s, e):
    limiter.wait() # STRICT WAIT
    try:
        data = {
            "symbol": symbol,
            "resolution": TIMEFRAME,
            "date_format": "1",
            "range_from": s.strftime("%Y-%m-%d"),
            "range_to": e.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        r = fyers.history(data=data)
        
        if r.get("s") == "ok":
            return r.get("candles", [])
        else:
            msg = str(r.get("message", "") or "").strip()
            msg_lower = msg.lower()
            
            if any(term in msg_lower for term in ["rate limit", "429", "too many requests", "request limit reached"]):
                raise Exception(f"Rate Limit Hit: {msg}")
            
            if msg and "no_data" not in msg_lower and "no data" not in msg_lower:
                with open(LOG_FILE, "a") as ef:
                    ef.write(f"{datetime.now()} | {symbol} | {s.date()} - {e.date()} | {msg}\n")
            return []
            
    except Exception as ex:
        with open(os.path.join(LOG_DIR, "api_errors.log"), "a") as ef:
            ef.write(f"{datetime.now()} | {symbol} | Critical: {str(ex)}\n")
        return []
# ==========================================
# MAIN
# ==========================================
def main(client_id, access_token, max_chunk_days, start_date, end_date, timeframe, universe_tag, data_dir, log_dir, log_file, force_new_file, existing_data_file, completed_symbols=None):
    if completed_symbols is None:
        completed_symbols = set()
    global CLIENT_ID, ACCESS_TOKEN, MAX_CHUNK_DAYS, START_DATE, END_DATE, TIMEFRAME
    global UNIVERSE_TAG, FORCE_NEW_FILE, DATA_DIR, LOG_DIR, LOG_FILE, UNIVERSE_SYMBOLS, fyers
    
    CLIENT_ID = client_id
    ACCESS_TOKEN = access_token
    MAX_CHUNK_DAYS = max_chunk_days
    START_DATE = start_date
    END_DATE = end_date
    TIMEFRAME = timeframe
    UNIVERSE_TAG = universe_tag
    FORCE_NEW_FILE = force_new_file
    DATA_DIR = data_dir
    LOG_DIR = log_dir
    LOG_FILE = log_file

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        import json
        UNIVERSE_SYMBOLS = json.load(open(f"{DATA_DIR}/{UNIVERSE_TAG}_symbols.json"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe file not found: {DATA_DIR}/{UNIVERSE_TAG}_symbols.json")

    fyers = fyersModel.FyersModel(
        client_id=CLIENT_ID,
        token=ACCESS_TOKEN,
        is_async=False,
        log_path=LOG_DIR
    )

    errors = []
    wrote_data = False

    req_s = datetime.strptime(START_DATE, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    req_e = datetime.strptime(END_DATE, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    existing = None if FORCE_NEW_FILE else find_existing_file(existing_data_file)
    
    if existing:
        print(f"Resuming from: {os.path.basename(existing)}")
        symbol_bounds = scan_symbol_bounds(existing)
        output_file = existing
    else:
        print("Creating new data file...")
        symbol_bounds = {}
        output_file = os.path.join(DATA_DIR, full_filename(START_DATE, END_DATE))
        with open(output_file, "w", newline="") as f:
            csv.writer(f).writerow(["Symbol","Epoch","Datetime","Open","High","Low","Close","Volume"])

    # 3. Processing Loop
    with open(output_file, "a", newline="") as out:
        writer = csv.writer(out)

        for stock, symbol in tqdm(UNIVERSE_SYMBOLS.items(), desc="Processing Symbols"):
            if symbol in completed_symbols:
                continue
            try:
                bounds = symbol_bounds.get(symbol)
                fetch_ranges = []

                if bounds is None:
                    fetch_ranges = split_ranges(req_s, req_e)
                else:
                    file_min_dt = bounds[0]
                    file_max_dt = bounds[1]

                    if req_s.date() < file_min_dt.date():
                        backfill_end = (file_min_dt - timedelta(days=1)).replace(hour=23, minute=59, second=59)
                        fetch_ranges += split_ranges(req_s, backfill_end)

                    if req_e.date() > file_max_dt.date():
                        forward_start = (file_max_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                        if forward_start <= req_e:
                            fetch_ranges += split_ranges(forward_start, req_e)

                if not fetch_ranges:
                    continue

                for s, e in fetch_ranges:
                    candles = fetch_history(symbol, s, e)
                    if candles:
                        for c in candles:
                            dt_str = datetime.fromtimestamp(c[0]).strftime("%Y-%m-%d %H:%M:%S")
                            writer.writerow([symbol, c[0], dt_str, c[1], c[2], c[3], c[4], c[5]])
                        wrote_data = True

                completed_symbols.add(symbol)
                        
            except Exception as ex:
                errors.append((stock, str(ex)))

    # 4. Final Cleanup (Sort/Rename)
    final_min_ep = None
    final_max_ep = None

    if wrote_data or existing:
        final_min_ep, final_max_ep = sort_csv(output_file)
        
        if final_min_ep and final_max_ep:
            min_date_str = datetime.fromtimestamp(final_min_ep).strftime("%Y%m%d")
            max_date_str = datetime.fromtimestamp(final_max_ep).strftime("%Y%m%d")
            
            new_filename = f"{base_filename()}_{min_date_str}_{max_date_str}.csv"
            new_filepath = os.path.join(DATA_DIR, new_filename)
            
            if new_filepath != output_file:
                print(f"Renaming file to: {new_filename}")
                os.replace(output_file, new_filepath)
                output_file = new_filepath
                
                # Update the .env file immediately
                update_env_file("DATA_FILE", new_filename)
                print(f"Updated .env with DATA_FILE={new_filename}")
    
    # 5. Summary
    print("\n" + "="*30)
    print("EXECUTION COMPLETE")
    print(f"Final File: {os.path.basename(output_file)}")
    
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            count = sum(1 for _ in f) - 1
        print(f"Total Rows: {count}")
    
    if errors:
        print(f"\nExecution Errors ({len(errors)}):")
        for s, e in errors:
            print(f"- {s}: {e}")

    gc.collect()
    return output_file, completed_symbols, wrote_data

if __name__ == "__main__":
    # Provides fallback defaults if run independently
    main("2026-02-03", "2026-03-01", "5", "nifty50", "./data", "./logs")