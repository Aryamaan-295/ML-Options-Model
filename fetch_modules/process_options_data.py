import pandas as pd
import numpy as np
import os
import sys
import logging
import warnings
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    class SettingWithCopyWarning(Warning): pass

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================
# 2. DATA MERGING & LOADING
# ==========================================

def load_and_merge_data(options_file, underlying_symbol):
    print(f"--- Loading Spot Data from {STOCKS_FILE} ---")
    
    if not os.path.exists(STOCKS_FILE):
        raise FileNotFoundError(f"Stocks file missing: {STOCKS_FILE}")

    # 1. Efficient Stock Load (Only needed columns)
    spots_list = []
    use_cols = ['Datetime', 'Symbol', 'Close']
    
    try:
        for chunk in pd.read_csv(STOCKS_FILE, usecols=use_cols, chunksize=CHUNK_SIZE):
            mask = chunk['Symbol'].isin([underlying_symbol, VIX_SYMBOL])
            if mask.any():
                c = chunk.loc[mask].copy()
                c['Datetime'] = pd.to_datetime(c['Datetime'])
                c['Date'] = c['Datetime'].dt.date
                spots_list.append(c)
    except Exception as e:
        logging.error(f"Stock Load Error: {e}")
        raise

    if not spots_list:
        raise ValueError("No matching stock data found.")

    full_spot = pd.concat(spots_list, ignore_index=True)
    full_spot.sort_values(['Date', 'Datetime'], inplace=True)
    
    # Pivot to get SpotPrice and IndiaVIX columns
    daily_close = full_spot.groupby(['Date', 'Symbol'])['Close'].last().unstack()
    rename_map = {underlying_symbol: 'SpotPrice', VIX_SYMBOL: 'IndiaVIX'}
    daily_close.rename(columns=rename_map, inplace=True)
    
    # Fill VIX if missing
    if 'IndiaVIX' not in daily_close.columns:
        daily_close['IndiaVIX'] = 15.0
    daily_close['IndiaVIX'] = daily_close['IndiaVIX'].fillna(15.0)

    print(f"--- Loading Options Data from {options_file} ---")
    if not os.path.exists(options_file):
        raise FileNotFoundError(f"Options file missing: {options_file}")

    df_opt = pd.read_csv(options_file)
    
    # Drop old calculated columns to ensure a clean slate
    drop_cols = ['SpotPrice', 'IndiaVIX', 'TimeYear', 'IV', 'delta', 'gamma', 'theta', 'rho', 'vega', 
                 'Clean_IV', 'Clean_Delta', 'Clean_Gamma', 'Clean_Theta', 'Clean_Vega', 'Clean_Rho', 'Raw_IV']
    df_opt.drop(columns=[c for c in drop_cols if c in df_opt.columns], inplace=True)

    # Prepare Merge Keys
    df_opt['Datetime_Obj'] = pd.to_datetime(df_opt['Datetime'])
    df_opt['Date_Key'] = df_opt['Datetime_Obj'].dt.date
    
    # Merge Spot Data
    print("--- Merging Spot & Options ---")
    spot_reset = daily_close.reset_index()
    merged_df = df_opt.merge(spot_reset, left_on='Date_Key', right_on='Date', how='left')
    
    # Cleanup
    merged_df.drop(columns=['Date_Key', 'Date', 'Datetime_Obj'], inplace=True)
    
    # Check Data Integrity
    missing_spot = merged_df['SpotPrice'].isna()
    if missing_spot.any():
        logging.warning(f"Dropping {missing_spot.sum()} rows due to missing Spot Price.")
        merged_df.dropna(subset=['SpotPrice'], inplace=True)
        
    return merged_df

# ==========================================
# 3. PRE-PROCESSING
# ==========================================

def calculate_metadata_and_raw_iv(df):
    print("--- Calculating Metadata & Raw IV ---")
    
    df['Expiry_Obj'] = pd.to_datetime(df['Expiry'])
    df['Datetime_Obj'] = pd.to_datetime(df['Datetime'])
    
    # Time to Expiry (Annualized)
    df['TimeYear'] = (df['Expiry_Obj'] - df['Datetime_Obj']).dt.days / 365.0
    # Handle 0 DTE or negative time (set to ~1 minute)
    df.loc[df['TimeYear'] <= 0.0001, 'TimeYear'] = 0.0001
    
    # Moneyness & Flags
    df['flag'] = df['Type'].map({
        'CE': 'c', 'PE': 'p', 
        'ce': 'c', 'pe': 'p',
        'C': 'c', 'P': 'p',
        'c': 'c', 'p': 'p'
    })
    # Strict Safety Check: Drop any rows where Type was invalid/missing to prevent math engine crashes
    df.dropna(subset=['flag'], inplace=True)
    
    # Log Moneyness (Standard for Vol Surfaces)
    df['Moneyness'] = np.log(df['Strike'] / df['SpotPrice'])

    # Vectorized Raw IV Calculation
    try:
        from py_vollib_vectorized import vectorized_implied_volatility
        df['Raw_IV'] = vectorized_implied_volatility(
            df['Close'].values, 
            df['SpotPrice'].values, 
            df['Strike'].values, 
            df['TimeYear'].values, 
            RISK_FREE_RATE, 
            df['flag'].values, 
            return_as='numpy',
            on_error='ignore' # Returns 0/NaN for bad inputs (like Price < Intrinsic)
        )
    except ImportError:
        print("CRITICAL ERROR: 'py_vollib_vectorized' is missing. Install via pip.")
        sys.exit(1)
        
    return df

# ==========================================
# 4. ROBUST SURFACE MODELING (The "Better Model")
# ==========================================

def fit_quadratic_smile(group):
    """
    Fits a Weighted Quadratic Smile: IV = a*k^2 + b*k + c
    (k = Log Moneyness)
    
    Improvement over Spline:
    - Enforces the parabolic "smile" shape (no wiggling).
    - Faster execution (Linear Algebra vs Iterative Solver).
    - Extrapolates safely to Deep ITM/OTM.
    """
    # 1. Define Anchors (Reliable Data Points)
    # We trust OTM/ATM options more than ITM options (due to liquidity/spreads)
    
    # Valid IV Range Filter
    valid_iv_mask = (group['Raw_IV'] > 0.01) & (group['Raw_IV'] < 2.0) & (group['Raw_IV'].notna())
    
    spot = group['SpotPrice'].iloc[0]
    
    # Strict OTM Definitions
    # Call OTM: Strike >= Spot
    # Put OTM: Strike <= Spot
    is_call = group['flag'] == 'c'
    is_put = group['flag'] == 'p'
    
    # We include a 5% buffer around ATM to ensure the curve connects in the middle
    otm_mask = (is_call & (group['Strike'] >= spot * 0.95)) | (is_put & (group['Strike'] <= spot * 1.05))
    
    anchor_mask = valid_iv_mask & otm_mask
    anchors = group.loc[anchor_mask]
    
    # FALLBACKS
    # If not enough OTM anchors, try using *any* valid IV (including ITM)
    if len(anchors) < 4:
        anchors = group.loc[valid_iv_mask]
        
    # If still not enough data, return a flat line (VIX or Median)
    if len(anchors) < 3:
        # Default to IndiaVIX or 0.2
        fallback_iv = group['IndiaVIX'].iloc[0] / 100.0
        if np.isnan(fallback_iv) or fallback_iv <= 0: fallback_iv = 0.2
        return pd.Series(fallback_iv, index=group.index)

    # 2. Fit the Quadratic Model (Polyfit degree 2)
    try:
        x = anchors['Moneyness'].values
        y = anchors['Raw_IV'].values
        
        # Weights: Give higher importance to options Near-The-Money (Moneyness ~ 0)
        # weight = 1 / (|k| + 0.1)
        weights = 1.0 / (np.abs(x) + 0.1)
        
        coeffs = np.polyfit(x, y, 2, w=weights)
        
        # 3. Predict for ALL strikes in this expiry (Fixing the bad ITM options)
        x_all = group['Moneyness'].values
        smooth_iv = np.polyval(coeffs, x_all)
        
        # 4. Clamp Results (Safety)
        # Prevents curve from shooting to infinity at extreme strikes
        smooth_iv = np.clip(smooth_iv, 0.05, 3.0)
        
        return pd.Series(smooth_iv, index=group.index)
        
    except Exception:
        # If math fails, fallback to VIX
        fallback_iv = group['IndiaVIX'].iloc[0] / 100.0
        return pd.Series(fallback_iv, index=group.index)

# ==========================================
# 5. SANITY CHECKS (Logs only IV issues)
# ==========================================

def perform_post_fit_sanity_check(df, underlying_symbol):
    logging.info(f"--- STARTING POST-FIT SANITY CHECK FOR {underlying_symbol} ---")
    
    # Check for Zero IVs (excluding expiring options)
    # 0.004 years is ~1.5 days. IV gets weird at expiry, so we ignore those errors.
    mask_zero = (df['Clean_IV'] <= 0.001) | (df['Clean_IV'].isna())
    mask_valid_time = df['TimeYear'] > 0.004
    
    bad_rows = df[mask_zero & mask_valid_time]
    
    if not bad_rows.empty:
        logging.warning(f"IV ISSUE FOR {underlying_symbol}: {len(bad_rows)} rows still have 0/NaN IV after cleaning.")
    else:
        logging.info(f"IV STATUS FOR {underlying_symbol}: Success. All non-expiry options have valid modeled IVs.")

# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def run_processing_pipeline(options_file, underlying_symbol):
    # 1. Load & Merge
    temp_file = options_file + ".tmp"
    df = load_and_merge_data(options_file, underlying_symbol)
    
    # 2. Calc Raw
    df = calculate_metadata_and_raw_iv(df)
    
    # 3. Surface Fitting
    print("--- Fitting Volatility Surface (Quadratic Model) ---")
    
    # Group by unique time-slice
    df['GroupID'] = df['Datetime'].astype(str) + "_" + df['Expiry'].astype(str)
    
    # Apply fit with Progress Bar
    tqdm.pandas(desc="Modeling Surfaces")
    df['Clean_IV'] = df.groupby('GroupID', group_keys=False).progress_apply(fit_quadratic_smile)
    
    # Fill any remaining NaNs (failsafes)
    df['Clean_IV'] = df['Clean_IV'].fillna(df['IndiaVIX'] / 100.0).fillna(0.2)
    
    # 4. Calculate Clean Greeks
    print("--- Recalculating Greeks with Clean IV ---")
    from py_vollib_vectorized import get_all_greeks
    
    clean_greeks = get_all_greeks(
        df['flag'].values, 
        df['SpotPrice'].values, 
        df['Strike'].values, 
        df['TimeYear'].values, 
        RISK_FREE_RATE, 
        df['Clean_IV'].values, # Use the Modeled IV
        model='black_scholes', 
        return_as='dataframe'
    )
    
    clean_greeks.rename(columns={
        'delta': 'Clean_Delta',
        'gamma': 'Clean_Gamma',
        'theta': 'Clean_Theta',
        'rho':   'Clean_Rho',
        'vega':  'Clean_Vega'
    }, inplace=True)
    
    # Merge Greeks back
    final_df = pd.concat([df, clean_greeks], axis=1)
    
    # 5. Sanity Check
    perform_post_fit_sanity_check(final_df, underlying_symbol)
    
    # 6. Final Column Selection & Save
    print(f"--- Saving Cleaned Data to {temp_file} ---")
    
    # Map Clean columns to Standard columns for final output
    final_df['IV'] = final_df['Clean_IV']
    final_df['delta'] = final_df['Clean_Delta']
    final_df['gamma'] = final_df['Clean_Gamma']
    final_df['theta'] = final_df['Clean_Theta']
    final_df['rho'] = final_df['Clean_Rho']
    final_df['vega'] = final_df['Clean_Vega']

    # 1. Map CE/PE to C/P to save characters
    if 'Type' in final_df.columns:
        final_df['Type'] = final_df['Type'].replace({'CE': 'C', 'PE': 'P', 'ce': 'C', 'pe': 'P'})
        
    # 2. Drop the redundant Symbol column (Since filename indicates underlying)
    if 'Symbol' in final_df.columns:
        final_df.drop(columns=['Symbol'], inplace=True)
        
    # 3. Typecast Strike to integer (removes the trailing .0 safely)
    if 'Strike' in final_df.columns:
        # We fillna(0) as a strict safety fallback so astype(int) never crashes the script
        final_df['Strike'] = final_df['Strike'].fillna(0).astype(int)
        
    # 4. Truncate trailing decimals of Greeks and IV to 6 places
    round_cols = ['IV', 'delta', 'gamma', 'theta', 'rho', 'vega']
    for c in round_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].round(6)
    
    cols_to_drop = [
        'Datetime_Obj', 
        'Expiry_Obj', 
        'GroupID', 
        'flag', 
        'Moneyness', 
        'Raw_IV', 
        'Clean_IV',     
        'Clean_Delta',  
        'Clean_Gamma',  
        'Clean_Theta',  
        'Clean_Rho',    
        'Clean_Vega',
        'TimeYear',
        'ChgOI',
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_drop_cols = [c for c in cols_to_drop if c in final_df.columns]
    final_df.drop(columns=existing_drop_cols, inplace=True)
    
    # Save everything else
    final_df.to_csv(temp_file, index=False)

    print(f"--- Finalizing File Swap for {underlying_symbol} ---")
    if os.path.exists(temp_file):
        if os.path.exists(options_file):
            os.remove(options_file)
        os.rename(temp_file, options_file)
        print(f"SUCCESS. File updated: {options_file}")
    else:
        print(f"ERROR: Temp file not found for {options_file}.")
    
    print("--- Done ---")

def main(data_dir, stocks_file, log_dir, log_file, vix_symbol, risk_free_rate, chunk_size):
    global DATA_DIR, STOCKS_FILE, LOG_DIR, LOG_FILE
    global VIX_SYMBOL, RISK_FREE_RATE, CHUNK_SIZE
    
    DATA_DIR = data_dir
    STOCKS_FILE = stocks_file
    LOG_DIR = log_dir
    LOG_FILE = log_file
    VIX_SYMBOL = vix_symbol
    RISK_FREE_RATE = risk_free_rate
    CHUNK_SIZE = chunk_size
    
    logging.basicConfig(
        filename=LOG_FILE,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        force=True 
    )

    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: DATA_DIR '{DATA_DIR}' does not exist.")
        return
    
    options_files = [f for f in os.listdir(DATA_DIR) if "_opt_" in f and f.endswith(".csv")]
    
    if not options_files:
        print(f"No options files found in {DATA_DIR} matching pattern '*_opt_*.csv'.")
        return
        
    for file_name in options_files:
        underlying_symbol = "NSE:" + file_name.split('_opt_')[0]
        options_file_path = os.path.join(DATA_DIR, file_name)
        
        print(f"\n{'='*50}")
        print(f"Processing File: {file_name}")
        print(f"Detected Underlying: {underlying_symbol}")
        print(f"{'='*50}")
        
        try:
            run_processing_pipeline(options_file_path, underlying_symbol)
        except Exception as e:
            print(f"\nCRITICAL ERROR processing {file_name}: {e}")
            logging.exception(f"Fatal Execution Error for {file_name}")
            
            temp_file = options_file_path + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            print(f"Skipping {file_name} due to error. Moving to next file...")
            continue
            
    print(f"\nAll discovered files processed. Check logs at: {LOG_FILE}")

if __name__ == "__main__":
    main("./data", "./data/stocks_data.csv", "./logs")