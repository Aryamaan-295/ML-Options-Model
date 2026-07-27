#!/usr/bin/env python3
"""
NSE F&O + Commodity Options + Commodity Futures Data Fetcher
Fetches equity derivatives (NIFTY, BANKNIFTY), NSE commodity options bhavcopy,
and NSE commodity futures from the same commodity bhavcopy file.

All output folders (data/, temp/, log files) are created relative to the
directory you RUN this script from — not the directory the script lives in.

Usage:
    python nse_fetcher.py --start 2026-01-01 --end 2026-04-17
    python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --comm-tickers CRUDEOIL GOLD SILVER
    python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --no-commodities
    python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --only-commodities --comm-tickers CRUDEOIL
"""

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
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import xml.etree.ElementTree as ET

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# BASE DIR — always the directory you run the script from (project root)
# ==========================================
BASE_DIR = os.getcwd()

# ==========================================
# DEFAULT CONFIGURATION (overridden by CLI / main())
# ==========================================
DEFAULT_START_DATE        = date(2026, 1, 1)
DEFAULT_END_DATE          = date(2026, 4, 17)
DEFAULT_TICKERS           = ["NIFTY", "BANKNIFTY"]
DEFAULT_COMM_TICKERS      = []          # Empty list = fetch ALL commodity symbols
DEFAULT_FETCH_COMMODITIES = True
DEFAULT_FETCH_EQUITY      = True        # Set False via --only-commodities
DEFAULT_BATCH_SIZE        = 45

# All paths default to project root (CWD), not the script's own directory
DEFAULT_DATA_DIR          = os.path.join(BASE_DIR, "data")
DEFAULT_COMM_DATA_DIR     = os.path.join(BASE_DIR, "data", "commodities")
DEFAULT_LOG_DIR           = BASE_DIR    # Log files go directly into project root
DEFAULT_TEMP_DIR          = os.path.join(BASE_DIR, "temp", "price")
DEFAULT_TEMP_SPAN_DIR     = os.path.join(BASE_DIR, "temp", "span")
DEFAULT_TEMP_COMM_DIR     = os.path.join(BASE_DIR, "temp", "comm")

# NSE archive format change dates
DEFAULT_FORMAT_CHANGE_2024 = date(2024, 1, 1)
DEFAULT_FORMAT_CHANGE_2023 = date(2023, 1, 1)

# Equity F&O URLs
DEFAULT_URL_NEW  = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
DEFAULT_URL_OLD  = "https://archives.nseindia.com/content/historical/DERIVATIVES/{day:%Y}/{day:%b}/fo{day:%d%b%Y}bhav.csv.zip"
DEFAULT_API_URL  = "https://www.nseindia.com/api/reports"
DEFAULT_SPAN_URL = "https://archives.nseindia.com/content/nsccl/NSCCL_SPANFILE_{date}.zip"

# Commodity URL patterns — tried in order until one succeeds
# Correct path confirmed from NSE website: /content/com/
DEFAULT_COMM_URL_PATTERNS = [
    "https://nsearchives.nseindia.com/content/com/BhavCopy_NSE_CO_0_0_0_{date}_F_0000.csv.zip",
    "https://archives.nseindia.com/content/com/BhavCopy_NSE_CO_0_0_0_{date}_F_0000.csv.zip",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Accept": "*/*",
}

# ==========================================
# RUNTIME GLOBALS (injected by main())
# ==========================================
START_DATE = DEFAULT_START_DATE
END_DATE   = DEFAULT_END_DATE
TARGET_TICKERS        = DEFAULT_TICKERS
COMM_TICKERS          = DEFAULT_COMM_TICKERS
FETCH_COMMODITIES     = DEFAULT_FETCH_COMMODITIES
FETCH_EQUITY          = DEFAULT_FETCH_EQUITY
BATCH_SIZE            = DEFAULT_BATCH_SIZE
DATA_DIR              = DEFAULT_DATA_DIR
COMM_DATA_DIR         = DEFAULT_COMM_DATA_DIR
LOG_DIR               = DEFAULT_LOG_DIR
LOG_FILE              = os.path.join(DEFAULT_LOG_DIR, "nse_fetch.log")
COMM_LOG_FILE         = os.path.join(DEFAULT_LOG_DIR, "commodities_fetch.log")
TEMP_DIR              = DEFAULT_TEMP_DIR
TEMP_SPAN_DIR         = DEFAULT_TEMP_SPAN_DIR
TEMP_COMM_DIR         = DEFAULT_TEMP_COMM_DIR
FORMAT_CHANGE_DATE_2024 = DEFAULT_FORMAT_CHANGE_2024
FORMAT_CHANGE_DATE_2023 = DEFAULT_FORMAT_CHANGE_2023
URL_TEMPLATE_NEW      = DEFAULT_URL_NEW
URL_TEMPLATE_OLD      = DEFAULT_URL_OLD
API_URL               = DEFAULT_API_URL
SPAN_BASE_URL         = DEFAULT_SPAN_URL
COMM_URL_PATTERNS     = DEFAULT_COMM_URL_PATTERNS
HEADERS               = DEFAULT_HEADERS


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
        if dt < date(2024, 5, 1):    return 50
        elif dt < date(2024, 12, 1): return 25
        elif dt < date(2026, 1, 1):  return 75
        else:                        return 65
    elif symbol == "BANKNIFTY":
        if dt < date(2020, 1, 1):    return 20
        elif dt < date(2023, 7, 1):  return 25
        elif dt < date(2025, 1, 1):  return 15
        elif dt < date(2025, 7, 31): return 30
        elif dt < date(2026, 1, 1):  return 35
        else:                        return 30
    return 0


# ==========================================
# LOGGING
# ==========================================
def init_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n--- NSE Options Fetch Started: {datetime.now()} ---\n")

def init_comm_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(COMM_LOG_FILE, 'a') as f:
        f.write(f"\n--- NSE Commodities Fetch Started: {datetime.now()} ---\n")

def log_error(date_str, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] Date: {date_str} | Error: {message}\n")

def log_commodity_error(date_str, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(COMM_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] Date: {date_str} | Error: {message}\n")


# ==========================================
# TEMP DIRECTORY MANAGEMENT
# ==========================================
def clean_temp():
    for d in [TEMP_DIR, TEMP_SPAN_DIR, TEMP_COMM_DIR]:
        if os.path.exists(d):
            try: shutil.rmtree(d)
            except: pass
        os.makedirs(d, exist_ok=True)


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
# INTELLIGENT SCHEDULING — EQUITY
# ==========================================
def get_existing_file_path(symbol):
    file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
    pattern = os.path.join(DATA_DIR, f"{file_prefix}_opt_*.csv")
    files = glob.glob(pattern)
    return files[0] if files else None

def analyze_missing_dates():
    print("Analyzing missing equity dates...")
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
# INTELLIGENT SCHEDULING — COMMODITY
# ==========================================
def get_existing_comm_file_path(symbol):
    pattern = os.path.join(COMM_DATA_DIR, f"{symbol}_comm_opt_*.csv")
    files = glob.glob(pattern)
    return files[0] if files else None

def get_existing_comm_futures_file_path(symbol):
    pattern = os.path.join(COMM_DATA_DIR, f"{symbol}_comm_fut_*.csv")
    files = glob.glob(pattern)
    return files[0] if files else None

def analyze_missing_comm_dates():
    """
    A date is considered complete only when it appears in BOTH options AND futures files.
    This ensures re-fetching if either side is missing for a given day.
    """
    print("Analyzing missing commodity dates...")
    delta = END_DATE - START_DATE
    target_dates = {START_DATE + timedelta(days=i) for i in range(delta.days + 1)}
    target_dates = {d for d in target_dates if d.weekday() < 5}

    # Dates present in options files
    opt_existing = set()
    for fp in glob.glob(os.path.join(COMM_DATA_DIR, "*_comm_opt_*.csv")):
        try:
            df = pd.read_csv(fp, usecols=['Datetime'])
            opt_existing.update(pd.to_datetime(df['Datetime']).dt.date)
        except:
            pass

    # Dates present in futures files
    fut_existing = set()
    for fp in glob.glob(os.path.join(COMM_DATA_DIR, "*_comm_fut_*.csv")):
        try:
            df = pd.read_csv(fp, usecols=['Datetime'])
            fut_existing.update(pd.to_datetime(df['Datetime']).dt.date)
        except:
            pass

    # Only days fully present in both are considered done
    fully_present = opt_existing & fut_existing

    if fully_present:
        missing = target_dates - fully_present
        if missing:
            print(f"[COMM] {len(missing)} missing day(s) out of {len(target_dates)} target days.")
        else:
            print("[COMM] Commodity data is up to date.")
        return sorted(list(missing))

    print(f"[COMM] No existing data found. Fetching all {len(target_dates)} target days.")
    return sorted(list(target_dates))


# ==========================================
# PART 1: EQUITY PRICE DATA FETCH
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
        if len(response.content) < 1000: return None, "File too small (likely HTML error)"

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

        if 'TckrSymb' in df.columns and 'FinInstrmNm' not in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'OptnTp': 'Type', 'XpryDt': 'Expiry',
                'StrkPric': 'Strike', 'BizDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades', 'NewBrdLotQty': 'LotSize'
            }
            if 'BizDt' not in df.columns:
                if 'TradDt' in df.columns:  df['BizDt'] = df['TradDt']
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

        df = df.rename(columns=col_map)
        valid_cols = [c for c in col_map.values() if c in df.columns]
        for needed in ['LotSize', 'Trades']:
            if needed not in valid_cols: valid_cols.append(needed)
            if needed not in df.columns: df[needed] = 0
        df = df[valid_cols]

        for date_col in ['Datetime', 'Expiry']:
            if date_col in df.columns:
                df[date_col] = df[date_col].astype(str).str.strip()
                df[f'{date_col}_temp'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
                mask = df[f'{date_col}_temp'].isna()
                if mask.any():
                    df.loc[mask, f'{date_col}_temp'] = pd.to_datetime(
                        df.loc[mask, date_col].str.title(), format='%d-%b-%Y', errors='coerce'
                    )
                df[date_col] = df[f'{date_col}_temp'].dt.strftime('%Y-%m-%d')
                df.drop(columns=[f'{date_col}_temp'], inplace=True)
                if df[date_col].isna().any():
                    df = df.dropna(subset=[date_col])

        def apply_lot_size(row):
            if row['LotSize'] > 0: return row['LotSize']
            return get_historical_lot_size(row['Symbol'], row['Datetime'])

        df['LotSize'] = df.apply(apply_lot_size, axis=1)

        num_cols = ['Strike', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:               df[c] = 0

        final_order = ['Symbol', 'Type', 'Expiry', 'Strike', 'Datetime', 'Open', 'High', 'Low', 'Close',
                       'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
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

                for series in oop_pf:
                    if strip_namespace(series.tag) == 'series':
                        expiry_dt = "UNKNOWN"
                        for child in series:
                            if strip_namespace(child.tag) == 'pe':
                                try: expiry_dt = datetime.strptime(child.text, "%Y%m%d").strftime("%Y-%m-%d")
                                except: expiry_dt = child.text
                                break

                        for opt in series:
                            if strip_namespace(opt.tag) == 'opt':
                                row = {"Symbol": sym, "Expiry": expiry_dt,
                                       "Strike": 0.0, "Type": "", "span_margin": 0.0}
                                scenarios = []
                                for prop in opt:
                                    tag = strip_namespace(prop.tag)
                                    if tag == 'k':
                                        try: row['Strike'] = float(prop.text)
                                        except: pass
                                    if tag == 'o':
                                        t = prop.text
                                        if t == 'C':   row['Type'] = 'CE'
                                        elif t == 'P': row['Type'] = 'PE'
                                        else:          row['Type'] = t
                                    if tag == 'ra':
                                        for val in prop:
                                            if strip_namespace(val.tag) == 'a':
                                                try: scenarios.append(float(val.text))
                                                except: pass
                                if len(scenarios) >= 16:
                                    for i in range(16): row[f"s_{i+1}"] = scenarios[i]
                                    row['span_margin'] = max(scenarios)
                                    data.append(row)
    except:
        pass

    return pd.DataFrame(data) if data else pd.DataFrame()

def fetch_span_data(session, target_date):
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
# PART 3: COMMODITY DATA FETCH
# ==========================================
def download_and_extract_commodity(session, target_date):
    """
    Tries each URL pattern in COMM_URL_PATTERNS in order.
    Correct NSE path confirmed: /content/com/BhavCopy_NSE_CO_0_0_0_{date}_F_0000.csv.zip
    Returns (csv_path, None) on success, (None, error_msg) if all patterns fail.
    The same CSV contains both futures AND options rows — processed separately downstream.
    """
    date_str = target_date.strftime("%Y%m%d")
    failed_patterns = []

    for idx, url_template in enumerate(COMM_URL_PATTERNS):
        url = url_template.format(date=date_str)
        try:
            response = session.get(url, timeout=15, verify=False)

            if response.status_code == 404:
                failed_patterns.append(f"Pattern {idx+1}: 404 — {url}")
                continue
            if response.status_code == 403:
                return None, f"403 Forbidden — {url}"
            if response.status_code != 200:
                failed_patterns.append(f"Pattern {idx+1}: HTTP {response.status_code} — {url}")
                continue
            if len(response.content) < 500:
                failed_patterns.append(f"Pattern {idx+1}: Response too small — {url}")
                continue

            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_name = next((n for n in z.namelist() if n.lower().endswith('.csv')), None)
                    if not csv_name:
                        failed_patterns.append(f"Pattern {idx+1}: No CSV in zip — {url}")
                        continue
                    z.extract(csv_name, TEMP_COMM_DIR)
                    return os.path.join(TEMP_COMM_DIR, csv_name), None
            except zipfile.BadZipFile:
                failed_patterns.append(f"Pattern {idx+1}: Bad zip file — {url}")
                continue

        except Exception as e:
            failed_patterns.append(f"Pattern {idx+1}: {str(e)} — {url}")
            continue

    detail = "\n    ".join(failed_patterns) if failed_patterns else "No patterns attempted"
    return None, f"All {len(COMM_URL_PATTERNS)} URL pattern(s) failed:\n    {detail}"


def _parse_dates_inplace(df, date_cols):
    """Shared dual-format date parser (YYYY-MM-DD and DD-Mon-YYYY) used by all processors."""
    for date_col in date_cols:
        if date_col not in df.columns:
            continue
        df[date_col] = df[date_col].astype(str).str.strip()
        df[f'{date_col}_temp'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
        mask = df[f'{date_col}_temp'].isna()
        if mask.any():
            df.loc[mask, f'{date_col}_temp'] = pd.to_datetime(
                df.loc[mask, date_col].str.title(), format='%d-%b-%Y', errors='coerce'
            )
        df[date_col] = df[f'{date_col}_temp'].dt.strftime('%Y-%m-%d')
        df.drop(columns=[f'{date_col}_temp'], inplace=True)
        if df[date_col].isna().any():
            df.dropna(subset=[date_col], inplace=True)
    return df


def process_commodity_data(csv_path, target_date):
    """
    Extracts OPTIONS rows (CE / PE) from the NSE commodity bhavcopy.
    Futures rows are handled separately by process_commodity_futures_data().
    Filters to COMM_TICKERS if specified; keeps all symbols otherwise.

    Output file pattern: {SYMBOL}_comm_opt_{min_date}_{max_date}.csv
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Format A: New NSE format (TckrSymb, no FinInstrmNm)
        if 'TckrSymb' in df.columns and 'FinInstrmNm' not in df.columns and 'OptnTp' in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'OptnTp': 'Type', 'XpryDt': 'Expiry',
                'StrkPric': 'Strike', 'BizDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades', 'NewBrdLotQty': 'LotSize'
            }
            if 'BizDt' not in df.columns:
                if 'TradDt' in df.columns:  df['BizDt'] = df['TradDt']
                elif 'RptgDt' in df.columns: df['BizDt'] = df['RptgDt']
            if COMM_TICKERS:
                df = df[df['TckrSymb'].isin(COMM_TICKERS)].copy()
            df = df[(df['OptnTp'].isin(['CE', 'PE'])) & (df['StrkPric'] > 0)].copy()
            if 'NewBrdLotQty' not in df.columns: df['LotSize'] = 0

        # Format B: Old INSTRUMENT/SYMBOL format
        elif 'INSTRUMENT' in df.columns and 'SYMBOL' in df.columns:
            col_map = {
                'SYMBOL': 'Symbol', 'OPTION_TYP': 'Type', 'EXPIRY_DT': 'Expiry',
                'STRIKE_PR': 'Strike', 'TIMESTAMP': 'Datetime', 'OPEN': 'Open',
                'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close',
                'CONTRACTS': 'Volume', 'OPEN_INT': 'OI', 'CHG_IN_OI': 'ChgOI',
                'SETTLE_PR': 'SettlePrice'
            }
            if COMM_TICKERS:
                df = df[df['SYMBOL'].isin(COMM_TICKERS)].copy()
            if 'INSTRUMENT' in df.columns:
                df = df[df['INSTRUMENT'].str.contains('OPT', na=False)].copy()
            if 'OPTION_TYP' in df.columns:
                df = df[df['OPTION_TYP'].isin(['CE', 'PE'])].copy()
            df['Trades'] = 0
            df['LotSize'] = 0

        # Format C: TckrSymb with FinInstrmNm (intermediate NSE format)
        elif 'TckrSymb' in df.columns and 'FinInstrmNm' in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'OptnTp': 'Type', 'XpryDt': 'Expiry',
                'StrkPric': 'Strike', 'RptgDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades'
            }
            if COMM_TICKERS:
                df = df[df['TckrSymb'].isin(COMM_TICKERS)].copy()
            if 'OptnTp' in df.columns:
                df = df[(df['OptnTp'].isin(['CE', 'PE'])) & (df['StrkPric'] > 0)].copy()
            df['LotSize'] = 0

        else:
            return None, f"Unknown CSV Format. Cols: {list(df.columns[:8])}"

        if df.empty:
            return pd.DataFrame(), None

        df = df.rename(columns=col_map)
        valid_cols = [c for c in col_map.values() if c in df.columns]
        for needed in ['LotSize', 'Trades']:
            if needed not in valid_cols: valid_cols.append(needed)
            if needed not in df.columns: df[needed] = 0
        df = df[valid_cols]

        df = _parse_dates_inplace(df, ['Datetime', 'Expiry'])

        if 'Datetime' not in df.columns or df['Datetime'].isna().all():
            df['Datetime'] = target_date.strftime('%Y-%m-%d')

        num_cols = ['Strike', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI', 'ChgOI',
                    'SettlePrice', 'Trades', 'LotSize']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:               df[c] = 0

        final_order = ['Symbol', 'Type', 'Expiry', 'Strike', 'Datetime', 'Open', 'High', 'Low',
                       'Close', 'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
        for c in final_order:
            if c not in df.columns: df[c] = 0
        df = df[final_order]

        return df, None

    except Exception as e:
        return None, f"Processing Error: {e}"


def process_commodity_futures_data(csv_path, target_date):
    """
    Extracts FUTURES rows from the same NSE commodity bhavcopy CSV file.

    Row identification per format:
      Format A (TckrSymb, no FinInstrmNm):
          Futures rows have OptnTp NOT in ['CE','PE'] (blank/NaN/'XX') AND StrkPric == 0.
      Format B (INSTRUMENT/SYMBOL):
          Rows where INSTRUMENT contains 'FUT' (e.g. FUTCOM, FUTIRT).
      Format C (TckrSymb + FinInstrmNm):
          Rows where OptnTp is not 'CE' or 'PE'.

    Output columns : Symbol, Expiry, Datetime, Open, High, Low, Close,
                     Volume, OI, ChgOI, SettlePrice, Trades, LotSize
    Output file    : {SYMBOL}_comm_fut_{min_date}_{max_date}.csv
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # ---- Format A: New NSE format (TckrSymb, no FinInstrmNm) ----
        if 'TckrSymb' in df.columns and 'FinInstrmNm' not in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'XpryDt': 'Expiry',
                'BizDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades', 'NewBrdLotQty': 'LotSize'
            }
            if 'BizDt' not in df.columns:
                if 'TradDt' in df.columns:  df['BizDt'] = df['TradDt']
                elif 'RptgDt' in df.columns: df['BizDt'] = df['RptgDt']
            if COMM_TICKERS:
                df = df[df['TckrSymb'].isin(COMM_TICKERS)].copy()
            # Futures = rows that are NOT CE/PE options
            if 'OptnTp' in df.columns:
                df = df[~df['OptnTp'].isin(['CE', 'PE'])].copy()
            # Exclude any leftover rows that still carry a non-zero strike (option remnants)
            if 'StrkPric' in df.columns:
                df = df[pd.to_numeric(df['StrkPric'], errors='coerce').fillna(0) == 0].copy()
            if 'NewBrdLotQty' not in df.columns:
                df['LotSize'] = 0

        # ---- Format B: Old INSTRUMENT/SYMBOL format ----
        elif 'INSTRUMENT' in df.columns and 'SYMBOL' in df.columns:
            col_map = {
                'SYMBOL': 'Symbol', 'EXPIRY_DT': 'Expiry',
                'TIMESTAMP': 'Datetime', 'OPEN': 'Open',
                'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close',
                'CONTRACTS': 'Volume', 'OPEN_INT': 'OI', 'CHG_IN_OI': 'ChgOI',
                'SETTLE_PR': 'SettlePrice'
            }
            if COMM_TICKERS:
                df = df[df['SYMBOL'].isin(COMM_TICKERS)].copy()
            # Futures rows: INSTRUMENT contains 'FUT' (e.g. FUTCOM, FUTIRT)
            if 'INSTRUMENT' in df.columns:
                df = df[df['INSTRUMENT'].str.contains('FUT', na=False)].copy()
            df['Trades'] = 0
            df['LotSize'] = 0

        # ---- Format C: TckrSymb with FinInstrmNm (intermediate NSE format) ----
        elif 'TckrSymb' in df.columns and 'FinInstrmNm' in df.columns:
            col_map = {
                'TckrSymb': 'Symbol', 'XpryDt': 'Expiry',
                'RptgDt': 'Datetime', 'OpnPric': 'Open',
                'HghPric': 'High', 'LwPric': 'Low', 'ClsPric': 'Close',
                'TtlTradgVol': 'Volume', 'OpnIntrst': 'OI', 'ChngInOpnIntrst': 'ChgOI',
                'SttlmPric': 'SettlePrice', 'TtlNbOfTxsExctd': 'Trades'
            }
            if COMM_TICKERS:
                df = df[df['TckrSymb'].isin(COMM_TICKERS)].copy()
            if 'OptnTp' in df.columns:
                df = df[~df['OptnTp'].isin(['CE', 'PE'])].copy()
            df['LotSize'] = 0

        else:
            return None, f"Unknown CSV Format. Cols: {list(df.columns[:8])}"

        if df.empty:
            return pd.DataFrame(), None

        # ---- Rename & subset ----
        df = df.rename(columns=col_map)
        valid_cols = [c for c in col_map.values() if c in df.columns]
        for needed in ['LotSize', 'Trades']:
            if needed not in valid_cols: valid_cols.append(needed)
            if needed not in df.columns: df[needed] = 0
        df = df[valid_cols]

        # ---- Date parsing ----
        df = _parse_dates_inplace(df, ['Datetime', 'Expiry'])

        if 'Datetime' not in df.columns or df['Datetime'].isna().all():
            df['Datetime'] = target_date.strftime('%Y-%m-%d')

        # ---- Numerics ----
        num_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI', 'ChgOI',
                    'SettlePrice', 'Trades', 'LotSize']
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:               df[c] = 0

        final_order = ['Symbol', 'Expiry', 'Datetime', 'Open', 'High', 'Low',
                       'Close', 'Volume', 'OI', 'ChgOI', 'SettlePrice', 'Trades', 'LotSize']
        for c in final_order:
            if c not in df.columns: df[c] = 0
        df = df[final_order]

        return df, None

    except Exception as e:
        return None, f"Futures Processing Error: {e}"


# ==========================================
# EQUITY FILE MANAGEMENT
# ==========================================
def append_to_files(batch_df):
    if batch_df.empty: return

    for symbol, group in batch_df.groupby('Symbol'):
        file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
        existing_path = get_existing_file_path(symbol)
        save_group = group.copy()
        if 'Symbol' in save_group.columns:
            save_group.drop(columns=['Symbol'], inplace=True)
        if 'Type' in save_group.columns:
            save_group['Type'] = save_group['Type'].replace({'CE': 'C', 'PE': 'P'})

        if existing_path:
            try:
                curr = pd.read_csv(existing_path, dtype={'Symbol': str, 'Type': str})
                if 'Symbol' in curr.columns: curr.drop(columns=['Symbol'], inplace=True)
                if 'Type'   in curr.columns: curr['Type'] = curr['Type'].replace({'CE': 'C', 'PE': 'P'})
                combined = pd.concat([curr, save_group], ignore_index=True)
                combined.to_csv(existing_path, index=False)
            except:
                save_group.to_csv(existing_path, index=False)
        else:
            temp_name = os.path.join(DATA_DIR, f"{file_prefix}_opt_temp.csv")
            save_group.to_csv(temp_name, index=False)

def finalize_files():
    print("\nFinalizing equity files...")
    for symbol in TARGET_TICKERS:
        file_prefix = {"NIFTY": "NIFTY50-INDEX", "BANKNIFTY": "NIFTYBANK-INDEX"}.get(symbol, symbol)
        file_path = get_existing_file_path(symbol)
        if not file_path:
            temp = os.path.join(DATA_DIR, f"{file_prefix}_opt_temp.csv")
            if os.path.exists(temp): file_path = temp
        if not file_path or not os.path.exists(file_path): continue

        try:
            df = pd.read_csv(file_path, dtype={'Symbol': str, 'Type': str})
            if df.empty: continue
            if 'Symbol' in df.columns: df.drop(columns=['Symbol'], inplace=True)
            if 'Type'   in df.columns: df['Type'] = df['Type'].replace({'CE': 'C', 'PE': 'P'})

            df.drop_duplicates(subset=['Datetime', 'Type', 'Expiry', 'Strike'], keep='last', inplace=True)
            df.sort_values(by=['Datetime', 'Expiry', 'Strike', 'Type'], ascending=True, inplace=True)

            min_d, max_d = df['Datetime'].min(), df['Datetime'].max()
            new_name = f"{file_prefix}_opt_{min_d}_{max_d}.csv"
            new_path = os.path.join(DATA_DIR, new_name)
            df.to_csv(new_path, index=False)

            if file_path != new_path:
                os.remove(file_path)
                print(f"  Updated:  {new_name}")
            else:
                print(f"  Verified: {new_name}")
        except Exception as e:
            print(f"  Error finalizing {symbol}: {e}")


# ==========================================
# COMMODITY OPTIONS FILE MANAGEMENT
# ==========================================
def append_to_comm_files(batch_df):
    if batch_df.empty: return

    for symbol, group in batch_df.groupby('Symbol'):
        existing_path = get_existing_comm_file_path(symbol)
        save_group = group.copy()
        if 'Symbol' in save_group.columns:
            save_group.drop(columns=['Symbol'], inplace=True)
        if 'Type' in save_group.columns:
            save_group['Type'] = save_group['Type'].replace({'CE': 'C', 'PE': 'P'})

        if existing_path:
            try:
                curr = pd.read_csv(existing_path, dtype={'Type': str})
                if 'Symbol' in curr.columns: curr.drop(columns=['Symbol'], inplace=True)
                if 'Type'   in curr.columns: curr['Type'] = curr['Type'].replace({'CE': 'C', 'PE': 'P'})
                combined = pd.concat([curr, save_group], ignore_index=True)
                combined.to_csv(existing_path, index=False)
            except:
                save_group.to_csv(existing_path, index=False)
        else:
            temp_name = os.path.join(COMM_DATA_DIR, f"{symbol}_comm_opt_temp.csv")
            save_group.to_csv(temp_name, index=False)

def finalize_comm_files():
    print("\nFinalizing commodity options files...")
    all_comm_files = glob.glob(os.path.join(COMM_DATA_DIR, "*_comm_opt*.csv"))
    if not all_comm_files:
        print("  No commodity options files to finalize.")
        return

    for file_path in all_comm_files:
        try:
            df = pd.read_csv(file_path, dtype={'Type': str})
            if df.empty: continue

            if 'Symbol' in df.columns: df.drop(columns=['Symbol'], inplace=True)
            if 'Type'   in df.columns: df['Type'] = df['Type'].replace({'CE': 'C', 'PE': 'P'})

            df.drop_duplicates(subset=['Datetime', 'Type', 'Expiry', 'Strike'], keep='last', inplace=True)
            df.sort_values(by=['Datetime', 'Expiry', 'Strike', 'Type'], ascending=True, inplace=True)

            min_d, max_d = df['Datetime'].min(), df['Datetime'].max()
            base = os.path.basename(file_path)
            symbol_prefix = base.split('_comm_opt')[0]
            new_name = f"{symbol_prefix}_comm_opt_{min_d}_{max_d}.csv"
            new_path = os.path.join(COMM_DATA_DIR, new_name)
            df.to_csv(new_path, index=False)

            if file_path != new_path:
                os.remove(file_path)
                print(f"  Updated:  {new_name}")
            else:
                print(f"  Verified: {new_name}")
        except Exception as e:
            print(f"  Error finalizing {file_path}: {e}")


# ==========================================
# COMMODITY FUTURES FILE MANAGEMENT
# ==========================================
def append_to_comm_futures_files(batch_df):
    """Appends commodity futures batch data to per-symbol _comm_fut_ CSV files."""
    if batch_df.empty: return

    for symbol, group in batch_df.groupby('Symbol'):
        existing_path = get_existing_comm_futures_file_path(symbol)
        save_group = group.copy()
        if 'Symbol' in save_group.columns:
            save_group.drop(columns=['Symbol'], inplace=True)

        if existing_path:
            try:
                curr = pd.read_csv(existing_path)
                if 'Symbol' in curr.columns: curr.drop(columns=['Symbol'], inplace=True)
                combined = pd.concat([curr, save_group], ignore_index=True)
                combined.to_csv(existing_path, index=False)
            except:
                save_group.to_csv(existing_path, index=False)
        else:
            temp_name = os.path.join(COMM_DATA_DIR, f"{symbol}_comm_fut_temp.csv")
            save_group.to_csv(temp_name, index=False)

def finalize_comm_futures_files():
    """Deduplicates, sorts, and renames all _comm_fut_ files with date range in the filename."""
    print("\nFinalizing commodity futures files...")
    all_fut_files = glob.glob(os.path.join(COMM_DATA_DIR, "*_comm_fut*.csv"))
    if not all_fut_files:
        print("  No commodity futures files to finalize.")
        return

    for file_path in all_fut_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty: continue

            if 'Symbol' in df.columns: df.drop(columns=['Symbol'], inplace=True)

            df.drop_duplicates(subset=['Datetime', 'Expiry'], keep='last', inplace=True)
            df.sort_values(by=['Datetime', 'Expiry'], ascending=True, inplace=True)

            min_d, max_d = df['Datetime'].min(), df['Datetime'].max()
            base = os.path.basename(file_path)
            symbol_prefix = base.split('_comm_fut')[0]
            new_name = f"{symbol_prefix}_comm_fut_{min_d}_{max_d}.csv"
            new_path = os.path.join(COMM_DATA_DIR, new_name)
            df.to_csv(new_path, index=False)

            if file_path != new_path:
                os.remove(file_path)
                print(f"  Updated:  {new_name}")
            else:
                print(f"  Verified: {new_name}")
        except Exception as e:
            print(f"  Error finalizing {file_path}: {e}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main(
    start_date, end_date,
    target_tickers, batch_size,
    data_dir, log_dir,
    comm_tickers=None,
    fetch_commodities=True,
    fetch_equity=True,
    comm_data_dir=None,
    temp_dir=None,
    temp_span_dir=None,
    temp_comm_dir=None,
    log_file=None,
    comm_log_file=None,
    format_change_2024=None,
    format_change_2023=None,
    url_new=None,
    url_old=None,
    api_url=None,
    span_url=None,
    comm_url_patterns=None,
    headers=None,
    lot_size_func=None,
):
    global START_DATE, END_DATE, TARGET_TICKERS, BATCH_SIZE
    global DATA_DIR, COMM_DATA_DIR, LOG_DIR, LOG_FILE, COMM_LOG_FILE
    global TEMP_DIR, TEMP_SPAN_DIR, TEMP_COMM_DIR
    global FORMAT_CHANGE_DATE_2024, FORMAT_CHANGE_DATE_2023
    global URL_TEMPLATE_NEW, URL_TEMPLATE_OLD, API_URL, SPAN_BASE_URL
    global COMM_URL_PATTERNS, HEADERS, COMM_TICKERS, FETCH_COMMODITIES, FETCH_EQUITY
    global get_historical_lot_size

    START_DATE        = start_date
    END_DATE          = end_date
    TARGET_TICKERS    = target_tickers
    BATCH_SIZE        = batch_size
    DATA_DIR          = data_dir
    COMM_DATA_DIR     = comm_data_dir  or os.path.join(data_dir, "commodities")
    LOG_DIR           = log_dir
    LOG_FILE          = log_file       or os.path.join(log_dir, "nse_fetch.log")
    COMM_LOG_FILE     = comm_log_file  or os.path.join(log_dir, "commodities_fetch.log")
    # Temp dirs sit beside data_dir at the project root level
    TEMP_DIR          = temp_dir       or os.path.join(os.path.dirname(data_dir), "temp", "price")
    TEMP_SPAN_DIR     = temp_span_dir  or os.path.join(os.path.dirname(data_dir), "temp", "span")
    TEMP_COMM_DIR     = temp_comm_dir  or os.path.join(os.path.dirname(data_dir), "temp", "comm")
    FORMAT_CHANGE_DATE_2024 = format_change_2024 or DEFAULT_FORMAT_CHANGE_2024
    FORMAT_CHANGE_DATE_2023 = format_change_2023 or DEFAULT_FORMAT_CHANGE_2023
    URL_TEMPLATE_NEW  = url_new        or DEFAULT_URL_NEW
    URL_TEMPLATE_OLD  = url_old        or DEFAULT_URL_OLD
    API_URL           = api_url        or DEFAULT_API_URL
    SPAN_BASE_URL     = span_url       or DEFAULT_SPAN_URL
    COMM_URL_PATTERNS = comm_url_patterns or DEFAULT_COMM_URL_PATTERNS
    HEADERS           = headers        or DEFAULT_HEADERS
    COMM_TICKERS      = comm_tickers   or []
    FETCH_COMMODITIES = fetch_commodities
    FETCH_EQUITY      = fetch_equity
    if lot_size_func:
        get_historical_lot_size = lot_size_func

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(COMM_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    clean_temp()

    if FETCH_EQUITY:
        init_log_file()
    if FETCH_COMMODITIES:
        init_comm_log_file()

    print("=" * 60)
    print("  NSE Data Fetcher — Equity F&O + Commodity Options & Futures")
    print("=" * 60)
    if FETCH_EQUITY:
        print(f"  Equity tickers   : {TARGET_TICKERS}")
    else:
        print("  Equity fetch     : DISABLED (--only-commodities)")
    if FETCH_COMMODITIES:
        comm_label = COMM_TICKERS if COMM_TICKERS else "ALL"
        print(f"  Commodity tickers: {comm_label}  [options + futures]")
    else:
        print("  Commodity fetch  : DISABLED")
    print(f"  Date range       : {START_DATE}  →  {END_DATE}")
    print(f"  Working root     : {BASE_DIR}")
    print()

    session = create_initialized_session()

    equity_dates = analyze_missing_dates() if FETCH_EQUITY else []
    comm_dates   = analyze_missing_comm_dates() if FETCH_COMMODITIES else []

    all_dates_needed = sorted(set(equity_dates) | set(comm_dates))

    if not all_dates_needed:
        print("All data is up to date.")
        if FETCH_EQUITY:
            finalize_files()
        if FETCH_COMMODITIES:
            finalize_comm_files()
            finalize_comm_futures_files()
        return False

    equity_set = set(equity_dates)
    comm_set   = set(comm_dates)

    parts = []
    if FETCH_EQUITY:      parts.append(f"Equity: {len(equity_dates)} day(s)")
    if FETCH_COMMODITIES: parts.append(f"Commodity: {len(comm_dates)} day(s)")
    parts.append(f"Total unique: {len(all_dates_needed)} day(s)")
    print(" | ".join(parts))

    span_cols = [f"s_{i+1}" for i in range(16)] + ['span_margin']
    new_data_fetched = False

    for i in range(0, len(all_dates_needed), BATCH_SIZE):
        batch_dates = all_dates_needed[i: i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}: {len(batch_dates)} day(s)")

        equity_batch   = []
        comm_opt_batch = []
        comm_fut_batch = []
        pbar = tqdm(batch_dates)

        for d in pbar:
            pbar.set_description(f"Fetching {d}")
            time.sleep(random.uniform(0.5, 1.5))

            # ---- Equity fetch (skipped when --only-commodities) ----
            if FETCH_EQUITY and d in equity_set:
                csv_path, error = download_and_extract_price(session, d)
                if error:
                    if "404" not in error:
                        log_error(str(d), f"Price: {error}")
                        if "403" in error: time.sleep(10)
                else:
                    daily_price_df, error = process_daily_data(csv_path)
                    if error:
                        log_error(str(d), f"Price Process: {error}")
                    elif daily_price_df is not None and not daily_price_df.empty:
                        daily_span_df = fetch_span_data(session, d)
                        if not daily_span_df.empty:
                            daily_span_df['Strike']  = daily_span_df['Strike'].astype(float)
                            daily_price_df['Strike'] = daily_price_df['Strike'].astype(float)
                            merged_df = pd.merge(daily_price_df, daily_span_df,
                                                 on=['Symbol', 'Type', 'Expiry', 'Strike'], how='left')
                            for col in span_cols:
                                merged_df[col] = merged_df[col].fillna(0) if col in merged_df.columns else 0
                        else:
                            merged_df = daily_price_df.copy()
                            for col in span_cols:
                                merged_df[col] = 0
                        equity_batch.append(merged_df)

            # ---- Commodity fetch — one download, two processors ----
            if FETCH_COMMODITIES and d in comm_set:
                comm_csv_path, comm_error = download_and_extract_commodity(session, d)
                if comm_error:
                    log_commodity_error(str(d), comm_error)
                    if "403" in comm_error: time.sleep(10)
                else:
                    # --- Options (CE/PE rows) ---
                    daily_comm_df, comm_proc_error = process_commodity_data(comm_csv_path, d)
                    if comm_proc_error:
                        log_commodity_error(str(d), f"Options Process: {comm_proc_error}")
                    elif daily_comm_df is not None and not daily_comm_df.empty:
                        comm_opt_batch.append(daily_comm_df)

                    # --- Futures (non-CE/PE rows, same CSV, no extra download) ---
                    daily_fut_df, fut_proc_error = process_commodity_futures_data(comm_csv_path, d)
                    if fut_proc_error:
                        log_commodity_error(str(d), f"Futures Process: {fut_proc_error}")
                    elif daily_fut_df is not None and not daily_fut_df.empty:
                        comm_fut_batch.append(daily_fut_df)

            clean_temp()

        # ---- Write equity batch ----
        if equity_batch:
            print(f"  → Writing {len(equity_batch)} equity day(s) to disk...")
            full_equity = pd.concat(equity_batch, ignore_index=True)
            append_to_files(full_equity)
            new_data_fetched = True
            del full_equity, equity_batch
            gc.collect()
        elif FETCH_EQUITY:
            print("  → No valid equity data in this batch.")

        # ---- Write commodity options batch ----
        if comm_opt_batch:
            print(f"  → Writing {len(comm_opt_batch)} commodity options day(s) to disk...")
            full_comm_opt = pd.concat(comm_opt_batch, ignore_index=True)
            append_to_comm_files(full_comm_opt)
            new_data_fetched = True
            del full_comm_opt, comm_opt_batch
            gc.collect()
        elif FETCH_COMMODITIES:
            print("  → No valid commodity options data in this batch.")

        # ---- Write commodity futures batch ----
        if comm_fut_batch:
            print(f"  → Writing {len(comm_fut_batch)} commodity futures day(s) to disk...")
            full_comm_fut = pd.concat(comm_fut_batch, ignore_index=True)
            append_to_comm_futures_files(full_comm_fut)
            new_data_fetched = True
            del full_comm_fut, comm_fut_batch
            gc.collect()
        elif FETCH_COMMODITIES:
            print("  → No valid commodity futures data in this batch.")

    if FETCH_EQUITY:
        finalize_files()
    if FETCH_COMMODITIES:
        finalize_comm_files()
        finalize_comm_futures_files()

    print(f"\nDone.")
    print(f"  Data dir       : {DATA_DIR}")
    if FETCH_COMMODITIES:
        print(f"  Commodity dir  : {COMM_DATA_DIR}")
        print(f"    Options files: *_comm_opt_*.csv")
        print(f"    Futures files: *_comm_fut_*.csv")
    if FETCH_EQUITY:
        print(f"  Equity log     : {LOG_FILE}")
    if FETCH_COMMODITIES:
        print(f"  Commodity log  : {COMM_LOG_FILE}")
    return new_data_fetched


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE F&O + Commodity Options & Futures Bhavcopy Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch equity (NIFTY + BANKNIFTY) and ALL commodity symbols (options + futures)
  python nse_fetcher.py --start 2026-01-01 --end 2026-04-17

  # Fetch CRUDEOIL commodity options + futures only (skip all equity F&O)
  python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --only-commodities --comm-tickers CRUDEOIL

  # Fetch specific commodity tickers alongside equity
  python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --comm-tickers CRUDEOIL GOLD SILVER

  # Skip commodity fetch entirely
  python nse_fetcher.py --start 2026-01-01 --end 2026-04-17 --no-commodities

  # Custom equity tickers and output directory
  python nse_fetcher.py --start 2025-01-01 --end 2025-12-31 \\
      --tickers NIFTY BANKNIFTY FINNIFTY \\
      --data-dir /data/nse
        """
    )
    parser.add_argument("--start",            type=str, default="2026-01-01",
                        help="Start date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--end",              type=str, default=date.today().isoformat(),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--tickers",          nargs="+", default=["NIFTY", "BANKNIFTY"],
                        help="Equity derivative tickers (default: NIFTY BANKNIFTY)")
    parser.add_argument("--comm-tickers",     nargs="+", default=[],
                        help="Commodity tickers to filter (default: empty = fetch ALL)")
    parser.add_argument("--no-commodities",   action="store_true",
                        help="Disable commodity data fetching entirely")
    parser.add_argument("--only-commodities", action="store_true",
                        help="Fetch commodity options + futures only — skip all equity F&O")
    parser.add_argument("--data-dir",         type=str, default=DEFAULT_DATA_DIR,
                        help="Output directory for equity data (default: <cwd>/data)")
    parser.add_argument("--comm-data-dir",    type=str, default=None,
                        help="Output directory for commodity data (default: <data-dir>/commodities)")
    parser.add_argument("--log-dir",          type=str, default=DEFAULT_LOG_DIR,
                        help="Directory for log files (default: cwd — project root)")
    parser.add_argument("--batch-size",       type=int, default=45,
                        help="Days per processing batch (default: 45)")

    args = parser.parse_args()

    if args.only_commodities and args.no_commodities:
        parser.error("--only-commodities and --no-commodities cannot be used together.")

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    if start_date > end_date:
        parser.error(f"--start ({args.start}) must be before --end ({args.end})")

    try:
        main(
            start_date=start_date,
            end_date=end_date,
            target_tickers=args.tickers,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            log_dir=args.log_dir,
            comm_tickers=args.comm_tickers,
            fetch_commodities=not args.no_commodities,
            fetch_equity=not args.only_commodities,
            comm_data_dir=args.comm_data_dir,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up temp files...")
        clean_temp()
    except Exception as e:
        print(f"\nFatal error: {e}")
        clean_temp()
        raise