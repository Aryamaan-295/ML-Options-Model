import glob
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import os

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
TARGET_INDEX = "NIFTY"  # Change this to "NIFTY50", "nifty", etc.

# Normalize the input to safely handle different casings and variations
_normalized_index = TARGET_INDEX.strip().upper()

if _normalized_index in ["NIFTY", "NIFTY50", "NIFTY 50"]:
    FILE_PREFIX = "nifty"
    SPOT_GLOB_PATTERN = "nifty50_5m_*.csv"
    OPT_GLOB_PATTERN = "NIFTY50-INDEX_opt_*.csv"
    SPOT_SYMBOL = "NSE:NIFTY50-INDEX"
else:
    # Easy placeholder to add BANKNIFTY or others in the future
    raise ValueError(f"Unsupported target index: {TARGET_INDEX}")

# ==========================================
# FILE PATH CONFIGURATION
# ==========================================
DATA_DIR = Path("./data")
PROCESSED_DIR = Path("./processed_data")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUT_FILE = PROCESSED_DIR / f"{FILE_PREFIX}_master_file.parquet"

# ==========================================
# PIPELINE FUNCTION
# ==========================================
def run_data_pipeline():
    print(f"\n{'='*50}")
    print(f"🚀 INITIALIZING {SPOT_SYMBOL} OPTIMIZED DATA PIPELINE")
    print(f"{'='*50}")

    # 1. Check for Incremental Updates
    max_date = None
    cutoff_date = None

    if MASTER_OUT_FILE.exists():
        max_date = pl.scan_parquet(MASTER_OUT_FILE).select(pl.col("Date").cast(pl.Date).max()).collect().item()
        if max_date is not None:
            cutoff_date = max_date - timedelta(days=35)
            print(f"[INIT] Existing dataset found. Last Date: {max_date}")
            print(f"[INIT] Incremental Mode Enabled. Processing raw data from: {cutoff_date}")
    else:
        print("[INIT] No existing dataset found. Processing full historical data.")

    # 2. Locate Raw Files
    global_files = glob.glob(str(DATA_DIR / "global_data_5m_*.csv"))
    spot_files   = glob.glob(str(DATA_DIR / SPOT_GLOB_PATTERN))
    opt_files    = glob.glob(str(DATA_DIR / OPT_GLOB_PATTERN))

    if not all([global_files, spot_files, opt_files]):
        raise FileNotFoundError("Missing one or more required CSV groups in ./data/")

    # ==========================================
    # 3. GLOBAL DATA (Macro Features)
    # ==========================================
    print("[PROCESS] Compiling Global Macro Features (UTC -> IST & 9:00 AM Alignment)...")
    lf_global = (
        pl.scan_csv(global_files)
        .with_columns([
            pl.col("timestamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Kolkata")
            .alias("timestamp_ist")
        ])
        .with_columns([
            pl.col("timestamp_ist").dt.date().alias("Date"),
            pl.col("timestamp_ist").dt.time().alias("Time")
        ])
    )

    if cutoff_date:
        lf_global = lf_global.filter(pl.col("Date") >= cutoff_date)

    df_global_daily = (
        lf_global
        .filter((pl.col("Time") >= pl.time(9, 0)) & (pl.col("Time") <= pl.time(15, 30)))
        .group_by(["symbol", "Date"])
        .agg([
            pl.col("open").filter(pl.col("Time") >= pl.time(9, 0)).first().alias("Open_0900"),
            pl.col("open").filter(pl.col("Time") >= pl.time(9, 15)).first().alias("Open_0915"),
            pl.col("close").filter(pl.col("Time") <= pl.time(15, 30)).last().alias("Close_1530")
        ])
        .sort(["symbol", "Date"])
        .with_columns([
            (pl.col("Close_1530") / pl.col("Open_0915")).log().alias("Intraday_Drift"),
            (pl.col("Open_0900").shift(-1) / pl.col("Close_1530")).log().over("symbol").alias("Overnight_Momentum")
        ])
        .collect()
    )

    lf_global_features = df_global_daily.pivot(
        index="Date",
        on="symbol",
        values=["Intraday_Drift", "Overnight_Momentum"]
    ).lazy()

    # ==========================================
    # 4. SPOT & VIX DATA
    # ==========================================
    print(f"[PROCESS] Compiling {FILE_PREFIX.upper()} Spot & VIX Daily Aggregations...")
    lf_spot_raw = (
        pl.scan_csv(spot_files)
        .with_columns([
            pl.col("Datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        ])
        .with_columns([pl.col("Datetime").dt.date().alias("Date")])
    )

    if cutoff_date:
        lf_spot_raw = lf_spot_raw.filter(pl.col("Date") >= cutoff_date)

    lf_index_daily = (
        lf_spot_raw.filter(pl.col("Symbol") == SPOT_SYMBOL)
        .sort("Datetime")
        .with_columns([
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return_5m"),
            (pl.col("Close") - pl.col("Close").shift(1)).abs().alias("abs_price_change_5m")
        ])
        .group_by("Date")
        .agg([
            pl.col("Open").first().alias("Spot_Open"),
            pl.col("High").max().alias("Spot_High"),
            pl.col("Low").min().alias("Spot_Low"),
            pl.col("Close").last().alias("Spot_Close"),
            pl.col("Volume").sum().alias("Spot_Volume"),
            (pl.col("log_return_5m").std() * ((75 * 252) ** 0.5)).alias("Realized_Vol_Ann"),
            pl.col("abs_price_change_5m").sum().alias("Path_Length")
        ])
        .with_columns([
            ((pl.col("Spot_Close") - pl.col("Spot_Open")).abs() /
              pl.when(pl.col("Path_Length") == 0).then(1.0).otherwise(pl.col("Path_Length"))
            ).alias("Trend_Strength")
        ])
    )

    lf_vix_daily = (
        lf_spot_raw.filter(pl.col("Symbol") == "NSE:INDIAVIX-INDEX")
        .sort("Datetime")
        .group_by("Date")
        .agg([pl.col("Close").last().alias("VIX_Close")])
    )

    # ==========================================
    # 5. OPTIONS DATA (PCR, 25-Delta Skew & GEX)
    # ==========================================
    print(f"[PROCESS] Compiling {FILE_PREFIX.upper()} Institutional Skew, PCR & GEX...")
    lf_opt = (
        pl.scan_csv(opt_files)
        .with_columns([
            pl.col("Datetime").str.strptime(pl.Date, "%Y-%m-%d").alias("Date"),
            pl.col("delta").cast(pl.Float64, strict=False),
            pl.col("IV").cast(pl.Float64, strict=False),
            pl.col("gamma").cast(pl.Float64, strict=False),
            pl.col("OI").cast(pl.Float64, strict=False),
            pl.col("LotSize").cast(pl.Float64, strict=False),
            pl.col("SpotPrice").cast(pl.Float64, strict=False),
        ])
    )

    if cutoff_date:
        lf_opt = lf_opt.filter(pl.col("Date") >= cutoff_date)

    lf_pcr = (
        lf_opt
        .group_by("Date")
        .agg([
            pl.col("Volume").filter(pl.col("Type").is_in(["C", "CE"])).sum().alias("Call_Volume"),
            pl.col("Volume").filter(pl.col("Type").is_in(["P", "PE"])).sum().alias("Put_Volume"),
            pl.col("OI").filter(pl.col("Type").is_in(["C", "CE"])).sum().alias("Call_OI"),
            pl.col("OI").filter(pl.col("Type").is_in(["P", "PE"])).sum().alias("Put_OI"),
        ])
        .with_columns([
            (pl.col("Put_OI") / pl.col("Call_OI").fill_null(1.0)).alias("PCR_OI"),
            (pl.col("Put_Volume") / pl.col("Call_Volume").fill_null(1.0)).alias("PCR_Volume")
        ])
    )

    lf_put_25d = (
        lf_opt.filter(pl.col("Type").is_in(["P", "PE"]))
        .drop_nulls(subset=["delta", "IV"])
        .with_columns((pl.col("delta").abs() - 0.25).abs().alias("delta_dist"))
        .sort(["Date", "delta_dist"])
        .group_by("Date").first()
        .select(["Date", pl.col("IV").alias("Put_IV_25d")])
    )

    lf_call_25d = (
        lf_opt.filter(pl.col("Type").is_in(["C", "CE"]))
        .drop_nulls(subset=["delta", "IV"])
        .with_columns((pl.col("delta").abs() - 0.25).abs().alias("delta_dist"))
        .sort(["Date", "delta_dist"])
        .group_by("Date").first()
        .select(["Date", pl.col("IV").alias("Call_IV_25d")])
    )

    lf_skew = lf_put_25d.join(lf_call_25d, on="Date", how="inner").with_columns(
        (pl.col("Put_IV_25d") - pl.col("Call_IV_25d")).alias("Options_Skew_25d")
    )

    # --- GEX Computation ---
    # GEX formula: Σ signed_gamma × OI × LotSize × SpotPrice² / 100
    # Sign convention: calls +gamma, puts -gamma.
    # Positive GEX → dealers long gamma → pin/vol suppression.
    # Negative GEX → dealers short gamma → move amplification.
    # GEX on day T reflects positioning carrying INTO day T+1 (no leakage).
    print(f"[PROCESS] Computing GEX (Gamma Exposure)...")
    lf_gex_raw = (
        lf_opt
        .drop_nulls(subset=["gamma", "OI", "LotSize", "SpotPrice"])
        .with_columns([
            pl.when(pl.col("Type").is_in(["C", "CE"]))
              .then(pl.col("gamma"))
              .otherwise(-pl.col("gamma"))
              .alias("signed_gamma")
        ])
        .with_columns([
            (
                pl.col("signed_gamma")
                * pl.col("OI")
                * pl.col("LotSize")
                * pl.col("SpotPrice").pow(2)
                / 100.0
            ).alias("gex_contrib")
        ])
        .group_by("Date")
        .agg(pl.col("gex_contrib").sum().alias("GEX_Raw"))
        .sort("Date")
    )

    # Rolling GEX features require a collect for stateful ops
    gex_df = lf_gex_raw.collect()
    gex_df = (
        gex_df
        .with_columns([
            pl.col("GEX_Raw").rolling_mean(window_size=20, min_samples=5).alias("_gex_mean"),
            pl.col("GEX_Raw").rolling_std(window_size=20, min_samples=5).alias("_gex_std"),
        ])
        .with_columns([
            ((pl.col("GEX_Raw") - pl.col("_gex_mean")) /
             (pl.col("_gex_std") + 1e-6)
            ).clip(-3.0, 3.0).alias("GEX_Zscore"),
        ])
        .with_columns([
            pl.col("GEX_Zscore").clip(lower_bound=0.0).alias("GEX_Pos"),
            (-pl.col("GEX_Zscore")).clip(lower_bound=0.0).alias("GEX_Neg"),
        ])
        .drop(["_gex_mean", "_gex_std"])
        .with_columns([
            pl.col("GEX_Raw").fill_null(strategy="forward"),
            pl.col("GEX_Zscore").fill_null(strategy="forward"),
            pl.col("GEX_Pos").fill_null(strategy="forward"),
            pl.col("GEX_Neg").fill_null(strategy="forward"),
        ])
        # FIX: guarantee pl.Date type on GEX before join
        .with_columns(pl.col("Date").cast(pl.Date))
    )
    lf_gex = gex_df.lazy()

    # ==========================================
    # 6. MASTER MERGE & TARGETS
    # ==========================================
    print("[MERGE] Unifying Datasets and Generating Forward Targets...")
    lf_master = (
        lf_index_daily
        .join(lf_vix_daily, on="Date", how="left")
        .join(lf_pcr, on="Date", how="left")
        .join(lf_skew, on="Date", how="left")
        .join(lf_gex, on="Date", how="left")
        .join(lf_global_features, on="Date", how="left")
        .sort("Date")
        .with_columns([
            (pl.col("Spot_Close").shift(-1) / pl.col("Spot_Close")).log().alias("Target_Ret_1D"),
            (pl.col("Spot_Close").shift(-5) / pl.col("Spot_Close")).log().alias("Target_Ret_5D"),
            ((pl.col("VIX_Close") / 100.0) - pl.col("Realized_Vol_Ann")).alias("VRP_Daily"),
            (pl.col("Spot_Close") / pl.col("Spot_Close").shift(5)).log().alias("Trailing_Ret_5D")
        ])
        .with_columns(pl.exclude("Date").fill_null(strategy="forward"))
        # FIX: enforce pl.Date after all joins so df_new schema is unambiguous
        .with_columns(pl.col("Date").cast(pl.Date))
    )

    df_new = lf_master.collect()

    # ==========================================
    # 7. COMBINE & ENGINEER ADVANCED FEATURES
    # ==========================================
    if max_date:
        print("[FILE IO] Merging new updates into Master File for feature engineering...")
        df_existing = pl.read_parquet(MASTER_OUT_FILE)
        # FIX: cast both sides to pl.Date before concat to prevent
        # Datetime('ms') vs Date schema conflicts caused by pandas roundtrip
        df_existing = df_existing.with_columns(pl.col("Date").cast(pl.Date))
        df_existing = df_existing.filter(pl.col("Date") < cutoff_date)
        df_new      = df_new.with_columns(pl.col("Date").cast(pl.Date))
        df_combined = pl.concat([df_existing, df_new], how="diagonal").sort("Date")
    else:
        df_combined = df_new.with_columns(pl.col("Date").cast(pl.Date))

    print("[PROCESS] Calculating Advanced Engineered Features...")
    df_pd = df_combined.to_pandas()

    # --- Start Exact Feature Engineering from Notebook ---
    df_pd['Vol_Change_1D']     = df_pd['Realized_Vol_Ann'].pct_change(1).clip(-2, 2)
    df_pd['Vol_Change_5D']     = df_pd['Realized_Vol_Ann'].pct_change(5).clip(-2, 2)
    df_pd['Vol_of_Vol_10D']    = df_pd['Realized_Vol_Ann'].rolling(10).std()
    df_pd['Recent_Max_AbsRet'] = df_pd['Target_Ret_1D'].shift(1).abs().rolling(5).max()

    for col in ['Overnight_Momentum_sp500', 'Overnight_Momentum_crudeoil', 'Overnight_Momentum_hang_seng']:
        if col in df_pd.columns:
            sign_col = np.sign(df_pd[col])
            df_pd[f'{col}_Consist_3D'] = sign_col.rolling(3).sum()
            df_pd[f'{col}_Consist_5D'] = sign_col.rolling(5).sum()
        else:
            df_pd[f'{col}_Consist_3D'] = 0.0
            df_pd[f'{col}_Consist_5D'] = 0.0

    df_pd['VoV_5D'] = df_pd['Realized_Vol_Ann'].rolling(5).std()
    df_pd['Vol_Regime_Zscore'] = (
        (df_pd['Realized_Vol_Ann'] - df_pd['Realized_Vol_Ann'].rolling(20).mean()) /
        (df_pd['Realized_Vol_Ann'].rolling(20).std() + 1e-6)
    ).clip(-3, 3)
    df_pd['Recent_Vol_Drawdown'] = (
        df_pd['Realized_Vol_Ann'] /
        (df_pd['Realized_Vol_Ann'].rolling(10).max() + 1e-6)
    ).clip(0.05, 1.5)
    df_pd['Vol_MA_Ratio'] = (
        df_pd['Realized_Vol_Ann'].rolling(5).mean() /
        (df_pd['Realized_Vol_Ann'].rolling(20).mean() + 1e-6)
    ).clip(0.1, 5.0)

    _close   = df_pd['Spot_Close']
    _open    = df_pd['Spot_Open']
    _high    = df_pd['Spot_High']
    _low     = df_pd['Spot_Low']
    _vol     = df_pd['Realized_Vol_Ann']
    _hist_lr = np.log(_close / _close.shift(1))

    _hl_range = np.log(_high / _low).clip(0, 0.10)
    df_pd['HL_Range_Zscore'] = (
        (_hl_range - _hl_range.rolling(20, min_periods=5).mean()) /
        (_hl_range.rolling(20, min_periods=5).std() + 1e-6)
    ).clip(-3, 3)
    df_pd['Overnight_Gap_Abs'] = np.log(_open / _close.shift(1)).abs().clip(0, 0.05)
    _upper_shadow = np.log(_high / np.maximum(_open, _close)).clip(0, 0.05)
    _lower_shadow = np.log(np.minimum(_open, _close) / _low).clip(0, 0.05)
    df_pd['Shadow_Imbalance'] = (_lower_shadow - _upper_shadow).clip(-0.05, 0.05)

    _down_ret = _hist_lr.where(_hist_lr < 0, 0.0)
    _up_ret   = _hist_lr.where(_hist_lr > 0, 0.0)
    df_pd['SemiVar_Down_10D'] = (
        _down_ret.rolling(10, min_periods=5).std() * np.sqrt(252)
    ).clip(0, 1.5).fillna(0)
    df_pd['SemiVar_Up_10D'] = (
        _up_ret.rolling(10, min_periods=5).std() * np.sqrt(252)
    ).clip(0, 1.5).fillna(0)
    df_pd['SemiVar_Ratio_10D'] = (
        df_pd['SemiVar_Down_10D'] / (df_pd['SemiVar_Up_10D'] + 1e-6)
    ).clip(0.1, 5.0)

    _below_mean = (_vol < _vol.rolling(20, min_periods=5).mean()).astype(int)
    _streak, _s = [], 0
    for v in _below_mean:
        _s = _s + 1 if v else 0
        _streak.append(_s)
    df_pd['Vol_Compression_Streak'] = np.log1p(np.array(_streak, dtype=np.float32))
    df_pd['Vol_Accel_3D'] = (_vol / (_vol.shift(3) + 1e-6)).clip(0.2, 5.0)

    df_pd['NIFTY_Overnight_Gap'] = np.log(_open / _close.shift(1)).clip(-0.05, 0.05)
    df_pd['NIFTY_Intraday_Ret']  = np.log(_close / _open).clip(-0.05, 0.05)
    _intra_sign = np.sign(df_pd['NIFTY_Intraday_Ret'])
    df_pd['NIFTY_Intraday_Consist_3D'] = _intra_sign.rolling(3).sum()
    df_pd['NIFTY_Intraday_Consist_5D'] = _intra_sign.rolling(5).sum()
    df_pd['PCR_OI_Change_1D'] = df_pd['PCR_OI'].pct_change(1).clip(-2, 2)
    df_pd['PCR_OI_Zscore'] = (
        (df_pd['PCR_OI'] - df_pd['PCR_OI'].rolling(20, min_periods=5).mean()) /
        (df_pd['PCR_OI'].rolling(20, min_periods=5).std() + 1e-6)
    ).clip(-3, 3)

    # --- DTE Expiry Regime Features ---
    # NIFTY weekly options expire on Thursdays (dayofweek == 3).
    _dow = pd.to_datetime(df_pd['Date']).dt.dayofweek
    df_pd['DTE_R0_Expiry']  = (_dow == 3).astype(float)  # Thursday  – expiry day
    df_pd['DTE_R1_PreExp']  = (_dow == 2).astype(float)  # Wednesday – day before expiry
    df_pd['DTE_R2_MidWeek'] = (_dow == 1).astype(float)  # Tuesday   – mid-week
    df_pd['DTE_R3_PostExp'] = (_dow == 4).astype(float)  # Friday    – day after expiry

    # --- Interaction Features ---
    _sp500_mom = df_pd.get('Overnight_Momentum_sp500', pd.Series(0.0, index=df_pd.index))
    df_pd['DTE_PostExp_x_SP500Mom'] = df_pd['DTE_R3_PostExp'] * _sp500_mom
    df_pd['GEX_x_DTE_R0'] = df_pd['GEX_Zscore'] * df_pd['DTE_R0_Expiry']

    df_pd.fillna(0, inplace=True)
    # --- End Feature Engineering ---

    # Convert back to Polars for saving
    df_final = pl.from_pandas(df_pd)

    # FIX: pandas always upcasts date -> datetime64[ns]; cast back to pl.Date
    # so the saved parquet schema stays consistent for future incremental runs
    df_final = df_final.with_columns(pl.col("Date").cast(pl.Date))

    # ==========================================
    # 8. SAVE OUTPUT
    # ==========================================
    df_final.write_parquet(MASTER_OUT_FILE)
    df_final.write_csv(PROCESSED_DIR / f"{FILE_PREFIX}_master_file.csv")

    print(f"\n[COMPLETE] Successfully saved to: {MASTER_OUT_FILE}")
    print(f"[COMPLETE] Final Dataset Shape: {df_final.shape}")

if __name__ == "__main__":
    run_data_pipeline()