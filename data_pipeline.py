import os
import sys
import shutil
import glob
import time
import logging
import pandas as pd
from datetime import date, datetime
from dotenv import load_dotenv, set_key

# ==========================================
# 1. ENVIRONMENT SETUP & SECRETS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(current_dir, ".env"))
load_dotenv(dotenv_path=env_path)

APP_ID = os.getenv("APP_ID")
SECRET_ID = os.getenv("SECRET_ID")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
AUTH_CODE = os.getenv("AUTH_CODE") 

DATA_DIR = os.getenv("DATA_DIR", os.path.abspath("./data"))
LOG_DIR = os.getenv("LOG_DIR", os.path.abspath("./logs"))
DATA_FILE = os.getenv("DATA_FILE") 
FULL_DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE) if DATA_FILE else None
MASTER_LOG_FILE = os.path.join(LOG_DIR, "data_pipeline.log")

# ==========================================
# 2. MASTER CONFIGURATIONS & CONSTANTS
# ==========================================
START_DATE = "2026-06-23"
END_DATE = "2026-07-30"

TIMEFRAME = "5"
UNIVERSE_TAG = "nifty50"
FORCE_NEW_FILE = False
MAX_CHUNK_DAYS = 90

TARGET_TICKERS = ["NIFTY", "BANKNIFTY"]
BATCH_SIZE = 45
TEMP_DIR = os.path.abspath("./temp_bhav_download")
TEMP_SPAN_DIR = os.path.abspath("./temp_span_download")
TEMP_PROCESS_DIR = os.path.join(DATA_DIR, "temp_process")

FORMAT_CHANGE_DATE_2024 = date(2024, 1, 1)
FORMAT_CHANGE_DATE_2023 = date(2023, 1, 1)

URL_TEMPLATE_NEW = "https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
URL_TEMPLATE_OLD = "https://archives.nseindia.com/content/fo/NSE_FO_bhavcopy_{day:%d%m%Y}.csv"
API_URL = "https://www.nseindia.com/api/reports"
SPAN_BASE_URL = "https://nsearchives.nseindia.com/archives/nsccl/span/nsccl.{date}.s.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/all-reports",
    "X-Requested-With": "XMLHttpRequest"
}

VIX_SYMBOL = "NSE:INDIAVIX-INDEX"
RISK_FREE_RATE = 0.06
CHUNK_SIZE_LIMIT = 200000

FINAL_COLUMNS = [
    "Type", "Expiry", "Strike", "Datetime", "Open", "High", "Low", "Close",
    "Volume", "OI", "SettlePrice", "Trades", "LotSize", "span_margin",
    "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_16",
    "IndiaVIX", "SpotPrice", "IV", "delta", "gamma", "theta", "rho", "vega"
]

# ==========================================
# 3. GLOBAL LOGIC & HELPERS
# ==========================================
def get_historical_lot_size(symbol, trade_date):
    if pd.isnull(trade_date): return 0
    try: dt = pd.to_datetime(trade_date).date()
    except: return 0 
        
    if symbol == "NIFTY":
        if dt < date(2024, 5, 1): return 50
        elif dt < date(2024, 12, 1): return 25
        elif dt < date(2026, 1, 1): return 75
        else: return 65 
    elif symbol == "BANKNIFTY":
        if dt < date(2020, 1, 1): return 20 
        elif dt < date(2023, 7, 1): return 25
        elif dt < date(2025, 1, 1): return 15
        elif dt < date(2025, 7, 31): return 30
        elif dt < date(2026, 1, 1): return 35
        else: return 30 
    return 0 

def cleanup_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                logging.info(f"🧹 Cleaned up temporary folder: {os.path.basename(directory)}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to clean up {directory}: {e}")

# ==========================================
# PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize the unified master logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(MASTER_LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("=== Pipeline Execution Started ===")

    logging.info("="*50)
    logging.info("STEP 1: GENERATING ACCESS TOKEN")
    logging.info("="*50)
    from info.login import generate_daily_access_token
    new_token = generate_daily_access_token()
    
    if new_token:
        set_key(env_path, "ACCESS_TOKEN", new_token)
        ACCESS_TOKEN = new_token
    elif not ACCESS_TOKEN:
        logging.error("❌ No valid Access Token available. Halting pipeline.")
        exit(1)

    logging.info("="*50)
    logging.info("STEP 2: FETCHING FYERS SPOT DATA")
    logging.info("="*50)
    from fetch_modules.data_fetch import main as fetch_spot_data
    
    final_full_path = None
    max_fyers_retries = 3
    completed_fyers_symbols = set()
    
    for attempt in range(max_fyers_retries):
        start_log_size = os.path.getsize(MASTER_LOG_FILE) if os.path.exists(MASTER_LOG_FILE) else 0
        rate_limit_hit = False
        
        try:
            final_full_path, completed_fyers_symbols, wrote_data = fetch_spot_data(
                client_id=APP_ID, access_token=ACCESS_TOKEN, max_chunk_days=MAX_CHUNK_DAYS,
                start_date=START_DATE, end_date=END_DATE, timeframe=TIMEFRAME, 
                universe_tag=UNIVERSE_TAG, data_dir=DATA_DIR, log_dir=LOG_DIR, 
                log_file=MASTER_LOG_FILE, force_new_file=FORCE_NEW_FILE,
                existing_data_file=FULL_DATA_FILE_PATH, completed_symbols=completed_fyers_symbols
            )
            
            FULL_DATA_FILE_PATH = final_full_path 
            
            if os.path.exists(MASTER_LOG_FILE):
                with open(MASTER_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(start_log_size)
                    new_logs = f.read().lower()
                    if any(term in new_logs for term in ["rate limit", "429", "too many requests", "request limit reached"]):
                        rate_limit_hit = True

            if rate_limit_hit:
                logging.warning(f"⚠️ Rate limit error discovered in log file.")
                if attempt < max_fyers_retries - 1:
                    logging.info(f"Sleeping for 5 seconds and retrying... (Attempt {attempt + 2}/{max_fyers_retries})")
                    time.sleep(5)
                    continue
                else:
                    logging.error("❌ Max retries reached for Fyers API. Pipeline will continue, but spot data may be incomplete.")
                    break
            else:
                break 
            
        except Exception as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["rate limit", "429", "too many requests", "request limit reached"]):
                logging.warning(f"⚠️ Rate limit error encountered: {e}")
                if attempt < max_fyers_retries - 1:
                    logging.info(f"Sleeping for 5 seconds and retrying... (Attempt {attempt + 2}/{max_fyers_retries})")
                    time.sleep(5)
                    continue
                else:
                    logging.error("❌ Max retries reached for Fyers API. Pipeline will continue, but spot data may be incomplete.")
                    break
            else:
                raise e
    
    if final_full_path:
        final_basename = os.path.basename(final_full_path)
        if final_basename != DATA_FILE:
            set_key(env_path, "DATA_FILE", final_basename)
            DATA_FILE = final_basename
            FULL_DATA_FILE_PATH = final_full_path
    
    logging.info("="*50)
    logging.info("STEP 3: FETCHING NSE OPTIONS DATA")
    logging.info("="*50)
    from fetch_modules.options_data_fetch import main as fetch_options_data
    options_data_found = fetch_options_data(
        start_date=date.fromisoformat(START_DATE), end_date=date.fromisoformat(END_DATE),
        target_tickers=TARGET_TICKERS, batch_size=BATCH_SIZE, data_dir=DATA_DIR,
        log_dir=LOG_DIR, log_file=MASTER_LOG_FILE, temp_dir=TEMP_DIR,
        temp_span_dir=TEMP_SPAN_DIR, format_change_2024=FORMAT_CHANGE_DATE_2024,
        format_change_2023=FORMAT_CHANGE_DATE_2023, url_new=URL_TEMPLATE_NEW,
        url_old=URL_TEMPLATE_OLD, api_url=API_URL, span_url=SPAN_BASE_URL,
        headers=HEADERS, lot_size_func=get_historical_lot_size
    )

    cleanup_directories([TEMP_DIR, TEMP_SPAN_DIR])

    if not options_data_found:
        logging.warning("="*50)
        logging.warning("🛑 No new NSE Options data fetched in this batch. Halting pipeline before Step 4.")
        logging.warning("="*50)
        exit(0)
    
    logging.info("="*50)
    logging.info("STEP 4: SMART PROCESSING OF OPTIONS & GREEKS")
    logging.info("="*50)
    from fetch_modules.process_options_data import main as process_options_data
    
    options_files = glob.glob(os.path.join(DATA_DIR, "*_opt_*.csv"))
    cleanup_directories([TEMP_PROCESS_DIR])
    os.makedirs(TEMP_PROCESS_DIR, exist_ok=True)
    files_to_merge = []
    
    for file_path in options_files:
        file_name = os.path.basename(file_path)
        logging.info(f"Analyzing {file_name} for missing Greeks...")
        
        df = pd.read_csv(file_path, dtype={'Type': str})
        
        if 'IV' not in df.columns or 'delta' not in df.columns:
            dates_to_process = df['Datetime'].unique()
        else:
            missing_mask = df['IV'].isnull() | df['delta'].isnull()
            dates_to_process = df.loc[missing_mask, 'Datetime'].unique()

        if len(dates_to_process) == 0:
            logging.info(f"  -> All dates have valid Greeks. Skipping.")
            continue
            
        logging.info(f"  -> Found {len(dates_to_process)} dates missing Greeks. Adding to batch queue...")
        subset_df = df[df['Datetime'].isin(dates_to_process)].copy()
        temp_file_path = os.path.join(TEMP_PROCESS_DIR, file_name)
        subset_df.to_csv(temp_file_path, index=False)
        
        files_to_merge.append({
            'original_path': file_path,
            'temp_path': temp_file_path,
            'missing_dates': dates_to_process
        })
        
    if files_to_merge:
        logging.info("-"*50)
        logging.info(f"Executing Batch Process Engine for {len(files_to_merge)} files...")
        logging.info("-"*50)
        try:
            process_options_data(
                data_dir=TEMP_PROCESS_DIR, stocks_file=FULL_DATA_FILE_PATH,
                log_dir=LOG_DIR, log_file=MASTER_LOG_FILE, vix_symbol=VIX_SYMBOL,
                risk_free_rate=RISK_FREE_RATE, chunk_size=CHUNK_SIZE_LIMIT
            )
            
            logging.info("Merging processed batch data back into master files...")
            for item in files_to_merge:
                orig_path = item['original_path']
                tmp_path = item['temp_path']
                dates = item['missing_dates']
                fname = os.path.basename(orig_path)
                
                if os.path.exists(tmp_path):
                    df_orig = pd.read_csv(orig_path, dtype={'Type': str})
                    processed_subset = pd.read_csv(tmp_path, dtype={'Type': str})
                    
                    df_unchanged = df_orig[~df_orig['Datetime'].isin(dates)].copy()
                    final_df = pd.concat([df_unchanged, processed_subset], ignore_index=True)
                    
                    for col in FINAL_COLUMNS:
                        if col not in final_df.columns:
                            final_df[col] = 0.0 
                            
                    final_df = final_df[FINAL_COLUMNS]
                    final_df.sort_values(by=['Datetime', 'Expiry', 'Strike', 'Type'], ascending=[True, True, True, True], inplace=True)
                    final_df.to_csv(orig_path, index=False)
                    logging.info(f"  -> Successfully merged new Greeks for {fname}.")
                else:
                    logging.warning(f"  -> ⚠️ Processing failed for {fname}. File missing.")
                    
        except Exception as e:
            logging.error(f"  -> ⚠️ Error during batch processing: {e}")

    cleanup_directories([TEMP_PROCESS_DIR])

    logging.info("="*50)
    logging.info("STEP 5: FETCHING GLOBAL DATA (DUKASCOPY)")
    logging.info("="*50)
    try:
        from fetch_modules.global_data_fetch import download_market_data as fetch_global_data
        dukascopy_timeframe = f"{TIMEFRAME}m" if str(TIMEFRAME).isdigit() else TIMEFRAME
        logging.info(f"Starting Global Data (Dukascopy) fetch for {START_DATE} to {END_DATE} at {dukascopy_timeframe}...")
        
        fetch_global_data(
            start_date=START_DATE, end_date=END_DATE, timeframe=dukascopy_timeframe
        )
    except Exception as e:
        logging.error(f"⚠️ Error during Global Data fetch: {e}")

    # =========================================================================
    # --- NEW ADDITION: STEP 6 SPREADS FETCH ---
    # =========================================================================
    logging.info("="*50)
    logging.info("STEP 6: FETCHING NIFTY SPREADS DATA")
    logging.info("="*50)
    try:
        from fetch_modules.spreads_fetch import run_fetch_pipeline as fetch_spreads_data
        logging.info("Starting Nifty Spreads Data fetch...")
        fetch_spreads_data()
    except Exception as e:
        logging.error(f"⚠️ Error during Spreads Data fetch: {e}")

    logging.info("="*50)
    logging.info("PIPELINE EXECUTION COMPLETE")
    logging.info("="*50)