import os
import io
import shutil
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm
import glob
import time
import random
import gc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import json
import xml.etree.ElementTree as ET

# Disable SSL Warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# LOT SIZE LOOKUP (Crash Proof)
# ==========================================
def get_historical_lot_size(symbol, trade_date):
    if pd.isnull(trade_date):
        return 0
        
    try:
        dt = pd.to_datetime(trade_date).date()
    except:
        return 0 
        
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

def init_log_file():
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n--- NSE Options Fetch Started: {datetime.now()} ---\n")

def log_error(date_str, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] Date: {date_str} | Error: {message}\n")

def clean_temp():
    # Clean Price Temp
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR)
        except: pass 
    # Clean SPAN Temp
    if os.path.exists(TEMP_SPAN_DIR):
        try: shutil.rmtree(TEMP_SPAN_DIR)
        except: pass
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(TEMP_SPAN_DIR, exist_ok=True)

# ==========================================
# ROBUST SESSION
# ==========================================
def create_initialized_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    
    try:
        session.get("https://www.nseindia.com", timeout=10, verify=False)
    except Exception as e:
        print(f"Cookie Init Warning: {e}")
    
    return session

# ==========================================
# INTELLIGENT SCHEDULING
# ==========================================
def get_existing_file_path(symbol):
    file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
    pattern = os.path.join(DATA_DIR, f"{file_prefix}_opt_*.csv")
    files = glob.glob(pattern)
    return files[0] if files else None

def analyze_missing_dates():
    print("Analyzing missing dates...")
    delta = END_DATE - START_DATE
    target_dates = {START_DATE + timedelta(days=i) for i in range(delta.days + 1)}
    target_dates = {d for d in target_dates if d.weekday() < 5} 
    
    global_dates_needed = set()
    
    for symbol in TARGET_TICKERS:
        file_path = get_existing_file_path(symbol)
        
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=['Datetime'])
                existing = set(pd.to_datetime(df['Datetime']).dt.date)
                missing = target_dates - existing
                
                if missing:
                    print(f"[{symbol}] File exists but missing {len(missing)} days.")
                    global_dates_needed.update(missing)
            except:
                print(f"[{symbol}] Error reading file. Re-fetching target range.")
                global_dates_needed.update(target_dates)
        else:
            global_dates_needed.update(target_dates)
            
    return sorted(list(global_dates_needed))

# ==========================================
# PART 1: PRICE DATA FETCH
# ==========================================
def download_and_extract_price(session, target_date):
    
    method = "ARCHIVE"
    if target_date < FORMAT_CHANGE_DATE_2023:
        method = "API"

    try:
        if method == "ARCHIVE":
            if target_date >= FORMAT_CHANGE_DATE_2024:
                url = URL_TEMPLATE_NEW.format(day=target_date)
                is_zip = True
            else:
                url = URL_TEMPLATE_OLD.format(day=target_date)
                is_zip = False
            response = session.get(url, timeout=15, verify=False)
        else:
            # API (Pre-2023)
            api_date_str = target_date.strftime("%d-%b-%Y") 
            params = {
                'archives': '[{"name":"F&O - Bhavcopy(csv)","type":"archives","category":"derivatives","section":"equity"}]',
                'date': api_date_str,
                'type': 'equity',
                'mode': 'single'
            }
            response = session.get(API_URL, params=params, timeout=20, verify=False)
            is_zip = True

        if response.status_code == 404: return None, "404 Not Found"
        if response.status_code == 403: return None, "403 Forbidden"
        if response.status_code != 200: return None, f"HTTP {response.status_code}"

        if len(response.content) < 1000:
             return None, "File too small (likely HTML error)"

        if is_zip:
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_name = next((n for n in z.namelist() if n.lower().endswith('.csv')), None)
                    if not csv_name: return None, "No CSV in Zip"
                    z.extract(csv_name, TEMP_DIR)
                    return os.path.join(TEMP_DIR, csv_name), None
            except zipfile.BadZipFile:
                return None, "Invalid Zip File"
        else:
            filename = f"NSE_FO_bhavcopy_{target_date.strftime('%d%m%Y')}.csv"
            file_path = os.path.join(TEMP_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path, None
            
    except Exception as e:
        return None, str(e)

def process_daily_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # COLUMNS MAPPING
        if 'TckrSymb' in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'OptnTp': 'Type', 'XpryDt': 'Expiry',
                'StrkPric': 'Strike', 'BizDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades', 'NewBrdLotQty': 'LotSize'
            }
            if 'BizDt' not in df.columns:
                if 'TradDt' in df.columns: df['BizDt'] = df['TradDt']
                elif 'RptgDt' in df.columns: df['BizDt'] = df['RptgDt']
            
            df = df[df['TckrSymb'].isin(TARGET_TICKERS)].copy()
            if 'OptnTp' in df.columns:
                df = df[(df['OptnTp'].isin(['CE', 'PE'])) & (df['StrkPric'] > 0)].copy()
            if 'NewBrdLotQty' not in df.columns: df['LotSize'] = 0

        elif 'INSTRUMENT' in df.columns and 'SYMBOL' in df.columns:
            col_map = {
                'SYMBOL': 'Symbol', 'OPTION_TYP': 'Type', 'EXPIRY_DT': 'Expiry',
                'STRIKE_PR': 'Strike', 'TIMESTAMP': 'Datetime', 'OPEN': 'Open',
                'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close',
                'CONTRACTS': 'Volume', 'OPEN_INT': 'OI', 'CHG_IN_OI': 'ChgOI',
                'SETTLE_PR': 'SettlePrice'
            }
            df = df[df['SYMBOL'].isin(TARGET_TICKERS)].copy()
            if 'INSTRUMENT' in df.columns:
                df = df[df['INSTRUMENT'] == 'OPTIDX'].copy()
            if 'OPTION_TYP' in df.columns:
                df = df[df['OPTION_TYP'].isin(['CE', 'PE'])].copy()
            df['Trades'] = 0 
            df['LotSize'] = 0 

        elif 'TckrSymb' in df.columns and 'FinInstrmNm' in df.columns:
             col_map = {
                'TckrSymb': 'Symbol', 'OptnTp': 'Type', 'XpryDt': 'Expiry',
                'StrkPric': 'Strike', 'RptgDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades'
            }
             df = df[df['TckrSymb'].isin(TARGET_TICKERS)].copy()
             if 'OptnTp' in df.columns:
                 df = df[(df['OptnTp'].isin(['CE', 'PE'])) & (df['StrkPric'] > 0)].copy()
             df['LotSize'] = 0

        else:
            return None, f"Unknown CSV Format. Cols: {list(df.columns[:5])}"

        if df.empty: return pd.DataFrame(), None

        # RENAME
        df = df.rename(columns=col_map)
        
        valid_cols = [c for c in col_map.values() if c in df.columns]
        for needed in ['LotSize', 'Trades']:
            if needed not in valid_cols: valid_cols.append(needed)
            if needed not in df.columns: df[needed] = 0
        df = df[valid_cols]

        # DATE PARSING
        for date_col in ['Datetime', 'Expiry']:
            if date_col in df.columns:
                df[date_col] = df[date_col].astype(str).str.strip()
                df[f'{date_col}_temp'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
                mask = df[f'{date_col}_temp'].isna()
                if mask.any():
                    df.loc[mask, f'{date_col}_temp'] = pd.to_datetime(
                        df.loc[mask, date_col].str.title(), 
                        format='%d-%b-%Y', 
                        errors='coerce'
                    )
                df[date_col] = df[f'{date_col}_temp'].dt.strftime('%Y-%m-%d')
                df.drop(columns=[f'{date_col}_temp'], inplace=True)
                if df[date_col].isna().any():
                    df = df.dropna(subset=[date_col])

        # LOT SIZE & NUMERICS
        def apply_lot_size(row):
            if row['LotSize'] > 0: return row['LotSize']
            return get_historical_lot_size(row['Symbol'], row['Datetime'])

        df['LotSize'] = df.apply(apply_lot_size, axis=1)

        num_cols = ['Strike', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = 0

        final_order = ['Symbol', 'Type', 'Expiry', 'Strike', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
        for c in final_order:
            if c not in df.columns: df[c] = 0
        df = df[final_order]

        return df, None
        
    except Exception as e:
        return None, f"Processing Error: {e}"

# ==========================================
# PART 2: SPAN DATA FETCH (Robust XML)
# ==========================================
def strip_namespace(tag):
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def parse_span_xml_to_df(filepath):
    """
    Parses SPAN XML and returns a DataFrame containing risk arrays for all target symbols.
    """
    data = []
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        for oop_pf in root.iter():
            if strip_namespace(oop_pf.tag) == 'oopPf':
                pf_code_elem = None
                for child in oop_pf:
                    if strip_namespace(child.tag) == 'pfCode':
                        pf_code_elem = child
                        break
                
                if pf_code_elem is None or pf_code_elem.text not in TARGET_TICKERS:
                    continue
                
                sym = pf_code_elem.text
                
                # Iterate Series (Expiries)
                for series in oop_pf:
                    if strip_namespace(series.tag) == 'series':
                        expiry_dt = "UNKNOWN"
                        for child in series:
                            if strip_namespace(child.tag) == 'pe':
                                try: expiry_dt = datetime.strptime(child.text, "%Y%m%d").strftime("%Y-%m-%d")
                                except: expiry_dt = child.text
                                break
                        
                        # Iterate Options
                        for opt in series:
                            if strip_namespace(opt.tag) == 'opt':
                                row = {
                                    "Symbol": sym,
                                    "Expiry": expiry_dt,
                                    "Strike": 0.0,
                                    "Type": "",
                                    "span_margin": 0.0
                                }
                                scenarios = []
                                for prop in opt:
                                    tag = strip_namespace(prop.tag)
                                    if tag == 'k': 
                                        try: row['Strike'] = float(prop.text)
                                        except: pass
                                    if tag == 'o': 
                                        # Normalize 'C' -> 'CE', 'P' -> 'PE' for matching
                                        t = prop.text
                                        if t == 'C': row['Type'] = 'CE'
                                        elif t == 'P': row['Type'] = 'PE'
                                        else: row['Type'] = t
                                    if tag == 'ra':
                                        for val in prop:
                                            if strip_namespace(val.tag) == 'a':
                                                try: scenarios.append(float(val.text))
                                                except: pass
                                
                                if len(scenarios) >= 16:
                                    for i in range(16): row[f"s_{i+1}"] = scenarios[i]
                                    row['span_margin'] = max(scenarios)
                                    data.append(row)
    except Exception as e:
        # Silently fail on parse error (return empty list)
        pass
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def fetch_span_data(session, target_date):
    """Downloads SPAN file and returns a DataFrame."""
    date_str = target_date.strftime("%Y%m%d")
    url = SPAN_BASE_URL.format(date=date_str)
    zip_path = os.path.join(TEMP_SPAN_DIR, f"nsccl.{date_str}.zip")
    
    try:
        resp = session.get(url, timeout=20, verify=False)
        if resp.status_code == 200:
            with open(zip_path, 'wb') as f: f.write(resp.content)
            try:
                with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(TEMP_SPAN_DIR)
            except: return pd.DataFrame()
            
            files = [f for f in glob.glob(os.path.join(TEMP_SPAN_DIR, "*")) if not f.endswith(".zip")]
            if files:
                return parse_span_xml_to_df(files[0])
    except:
        pass
    return pd.DataFrame()

# ==========================================
# FILE MANAGEMENT
# ==========================================
def append_to_files(batch_df):
    if batch_df.empty: return

    for symbol, group in batch_df.groupby('Symbol'):
        file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
        existing_path = get_existing_file_path(symbol)
        
        # Prepare the new data chunk
        save_group = group.copy()
        if 'Symbol' in save_group.columns:
            save_group.drop(columns=['Symbol'], inplace=True)
        if 'Type' in save_group.columns:
            save_group['Type'] = save_group['Type'].replace({'CE': 'C', 'PE': 'P'})
        
        if existing_path:
            try:
                # Specify dtype to prevent DtypeWarning on existing legacy files
                curr = pd.read_csv(existing_path, dtype={'Symbol': str, 'Type': str})
                
                # Clean up legacy columns in the existing file
                if 'Symbol' in curr.columns:
                    curr.drop(columns=['Symbol'], inplace=True)
                if 'Type' in curr.columns:
                    curr['Type'] = curr['Type'].replace({'CE': 'C', 'PE': 'P'})
                    
                combined = pd.concat([curr, save_group], ignore_index=True)
                combined.to_csv(existing_path, index=False)
            except Exception as e:
                save_group.to_csv(existing_path, index=False)
        else:
            temp_name = os.path.join(DATA_DIR, f"{file_prefix}_opt_temp.csv")
            save_group.to_csv(temp_name, index=False)

def finalize_files():
    print("\nFinalizing files...")
    for symbol in TARGET_TICKERS:
        file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
        file_path = get_existing_file_path(symbol)
        
        if not file_path:
            temp = os.path.join(DATA_DIR, f"{file_prefix}_opt_temp.csv")
            if os.path.exists(temp): file_path = temp
        
        if not file_path or not os.path.exists(file_path): continue

        try:
            # Specify dtype to prevent DtypeWarning on existing legacy files
            df = pd.read_csv(file_path, dtype={'Symbol': str, 'Type': str})
            if df.empty: continue
            
            # Final safety check to strip legacy formatting
            if 'Symbol' in df.columns:
                df.drop(columns=['Symbol'], inplace=True)
            if 'Type' in df.columns:
                df['Type'] = df['Type'].replace({'CE': 'C', 'PE': 'P'})
            
            # Deduplicate & Sort (Symbol removed from logic)
            df.drop_duplicates(subset=['Datetime', 'Type', 'Expiry', 'Strike'], keep='last', inplace=True)
            df.sort_values(by=['Datetime', 'Expiry', 'Strike', 'Type'], ascending=[True, True, True, True], inplace=True)
            
            min_d, max_d = df['Datetime'].min(), df['Datetime'].max()
            new_name = f"{file_prefix}_opt_{min_d}_{max_d}.csv"
            new_path = os.path.join(DATA_DIR, new_name)
            
            df.to_csv(new_path, index=False)
            
            if file_path != new_path: 
                os.remove(file_path)
                print(f"Updated: {new_name}")
            else:
                print(f"Verified: {new_name}")
                
        except Exception as e:
            print(f"Error finalizing {symbol}: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(start_date, end_date, target_tickers, batch_size, data_dir, log_dir, log_file, temp_dir, 
         temp_span_dir, format_change_2024, format_change_2023, url_new, url_old, 
         api_url, span_url, headers, lot_size_func):
         
    # Inject variables globally
    global START_DATE, END_DATE, TARGET_TICKERS, BATCH_SIZE, DATA_DIR, LOG_DIR, LOG_FILE
    global TEMP_DIR, TEMP_SPAN_DIR, FORMAT_CHANGE_DATE_2024, FORMAT_CHANGE_DATE_2023
    global URL_TEMPLATE_NEW, URL_TEMPLATE_OLD, API_URL, SPAN_BASE_URL, HEADERS
    global get_historical_lot_size
    
    START_DATE = start_date
    END_DATE = end_date
    TARGET_TICKERS = target_tickers
    BATCH_SIZE = batch_size
    DATA_DIR = data_dir
    LOG_DIR = log_dir
    LOG_FILE = log_file
    TEMP_DIR = temp_dir
    TEMP_SPAN_DIR = temp_span_dir
    FORMAT_CHANGE_DATE_2024 = format_change_2024
    FORMAT_CHANGE_DATE_2023 = format_change_2023
    URL_TEMPLATE_NEW = url_new
    URL_TEMPLATE_OLD = url_old
    API_URL = api_url
    SPAN_BASE_URL = span_url
    HEADERS = headers
    get_historical_lot_size = lot_size_func

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    clean_temp()
    init_log_file()
    print(f"--- Universal Data Fetcher (Price + SPAN Merged) ---")
    session = create_initialized_session()
    
    dates_to_download = analyze_missing_dates()
    if not dates_to_download:
        print("Data up to date.")
        finalize_files()
        return False

    print(f"Downloading {len(dates_to_download)} missing days...")
    new_data_fetched = False
    
    span_cols = [f"s_{i+1}" for i in range(16)] + ['span_margin']
    
    for i in range(0, len(dates_to_download), BATCH_SIZE):
        batch_dates = dates_to_download[i : i + BATCH_SIZE]
        print(f"\nBatch {i//BATCH_SIZE + 1}: {len(batch_dates)} days")
        
        batch_data = []
        pbar = tqdm(batch_dates)
        
        for d in pbar:
            pbar.set_description(f"Fetching {d}")
            time.sleep(random.uniform(0.5, 1.5))
            
            csv_path, error = download_and_extract_price(session, d)
            
            if error:
                if "404" not in error: 
                    log_error(str(d), f"Price: {error}")
                    if "403" in error: time.sleep(10)
                continue
            
            daily_price_df, error = process_daily_data(csv_path)
            
            if error:
                log_error(str(d), f"Price Process: {error}")
                continue
                
            if daily_price_df is None or daily_price_df.empty:
                continue

            daily_span_df = fetch_span_data(session, d)
            
            if not daily_span_df.empty:
                daily_span_df['Strike'] = daily_span_df['Strike'].astype(float)
                daily_price_df['Strike'] = daily_price_df['Strike'].astype(float)
                
                merged_df = pd.merge(
                    daily_price_df, 
                    daily_span_df, 
                    on=['Symbol', 'Type', 'Expiry', 'Strike'], 
                    how='left'
                )
                
                for col in span_cols:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].fillna(0)
                    else:
                        merged_df[col] = 0
            else:
                merged_df = daily_price_df.copy()
                for col in span_cols:
                    merged_df[col] = 0

            batch_data.append(merged_df)
            clean_temp()

        if batch_data:
            print(f"  -> Writing {len(batch_data)} valid days to disk...")
            full_batch = pd.concat(batch_data, ignore_index=True)
            append_to_files(full_batch)
            new_data_fetched = True
            del full_batch, batch_data
            gc.collect()
        else:
            print("  -> No valid data extracted in this batch.")

    finalize_files()
    print(f"\nDone. Check logs at: {LOG_FILE}")
    return new_data_fetched

if __name__ == "__main__":
    from datetime import date
    try: 
        main(date(2026, 2, 12), date(2026, 3, 1), ["NIFTY", "BANKNIFTY"], 45, "./data", "./logs")
    except KeyboardInterrupt: 
        clean_temp()
    except Exception as e: 
        print(e)