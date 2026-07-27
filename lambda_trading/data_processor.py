"""
Nifty Feature Pipeline — AWS Lambda
────────────────────────────────────
Root-cause fix: Lambda layers sometimes ship a vendored asyncio built for Python
3.4/3.5.  In those files `async` is used as a plain attribute name
(e.g. `tasks.async(...)`).  Python 3.7+ treats `async` as a reserved keyword and
raises a SyntaxError while *compiling* the file — before any user code runs.

The two-line block below removes every `/opt/…` path that shadows the interpreter's
own asyncio, so the correct stdlib copy is always imported first.
"""

# ── Asyncio layer-conflict fix (MUST be the very first executable lines) ──────
import sys as _sys, os as _os
_sys.path = [
    _p for _p in _sys.path
    if not (
        _p.startswith("/opt/")
        and _os.path.isfile(_os.path.join(_p, "asyncio", "base_events.py"))
    )
]
# ─────────────────────────────────────────────────────────────────────────────

import io, os, gc, json, math, lzma, struct, time, random, zipfile
import traceback, sys
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import boto3
import requests
import urllib3
import gridfs
from pymongo import MongoClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fyers_apiv3 import fyersModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(msg: str):
    print(msg, flush=True)


def _log_section(title: str):
    _log(f"\n{'═'*10} {title} {'═'*10}")


def _log_error(context: str, exc: Optional[Exception] = None):
    """Always print full traceback so nothing is silently swallowed."""
    _log(f"\n{'!'*10} ERROR in {context} {'!'*10}")
    if exc is not None:
        _log(f"Exception type : {type(exc).__name__}")
        _log(f"Exception value: {exc}")
    _log("Full traceback:")
    _log(traceback.format_exc())
    _log("!" * 40)


# ── Constants ─────────────────────────────────────────────────────────────────

RISK_FREE_RATE          = 0.06
VIX_SYMBOL              = "NSE:INDIAVIX-INDEX"
SPOT_SYMBOL             = "NSE:NIFTY50-INDEX"
TARGET_OPTION           = "NIFTY"
IST_OFFSET              = timedelta(hours=5, minutes=30)
FORMAT_CHANGE_DATE_2024 = date(2024, 1, 1)
FORMAT_CHANGE_DATE_2023 = date(2023, 1, 1)
NSE_URL_NEW             = "https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
NSE_URL_OLD             = "https://archives.nseindia.com/content/fo/NSE_FO_bhavcopy_{day:%d%m%Y}.csv"
NSE_API_URL             = "https://www.nseindia.com/api/reports"
NSE_HEADERS             = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":           "*/*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Connection":       "keep-alive",
    "Referer":          "https://www.nseindia.com/all-reports",
    "X-Requested-With": "XMLHttpRequest",
}
DUKA_SYMBOLS: Dict[str, str] = {
    "sp500":        "USA500IDXUSD",
    "dollar_index": "DOLLARIDXUSD",
    "hang_seng":    "HKGIDXHKD",
    "crudeoil":     "LIGHTCMDUSD",
}
_GC_PFX        = "_gc_"
_NIFTY_LO      = 5_000.0
_NIFTY_HI      = 45_000.0
_DB_NAME       = "deployments"
_GRIDFS_BUCKET = "processed_data"
_PARQUET_FNAME = "nifty_master_file.parquet"


# ── SSM ───────────────────────────────────────────────────────────────────────

def _ssm_get(name: str) -> str:
    client = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
    return client.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]


# ── MongoDB / GridFS ──────────────────────────────────────────────────────────

def _mongo_fs() -> Tuple[MongoClient, gridfs.GridFS]:
    uri    = _ssm_get("/fyers/MONGO_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=15_000)
    fs     = gridfs.GridFS(client[_DB_NAME], collection=_GRIDFS_BUCKET)
    return client, fs


def load_parquet_from_mongo() -> Optional[pd.DataFrame]:
    client, fs = _mongo_fs()
    try:
        files = list(fs.find({"filename": _PARQUET_FNAME}, sort=[("uploadDate", -1)], limit=1))
        if not files:
            _log("[MongoDB] No existing parquet found in GridFS.")
            return None
        gf  = fs.get(files[0]._id)
        buf = io.BytesIO(gf.read())
        gf.close()
        df  = pd.read_parquet(buf)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        _log(f"[MongoDB] Loaded {len(df)} rows, {len(df.columns)} columns. "
             f"Last date: {df['Date'].max()}")
        _log(f"[MongoDB] Existing columns ({len(df.columns)}): "
             f"{sorted(c for c in df.columns if not c.startswith(_GC_PFX))}")
        return df
    except Exception as e:
        _log_error("load_parquet_from_mongo", e)
        raise
    finally:
        client.close()


def save_parquet_to_mongo(df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    raw_bytes = buf.read()

    metadata = {
        "uploaded_at": datetime.utcnow().isoformat(),
        "rows":        len(df),
        "last_date":   str(df["Date"].max()),
        "source":      "nifty-feature-pipeline-lambda",
        "columns":     [c for c in df.columns if not c.startswith(_GC_PFX)],
    }

    tmp_name = f"__tmp__{_PARQUET_FNAME}"
    tmp_oid  = None
    client, fs = _mongo_fs()
    try:
        tmp_oid = fs.put(io.BytesIO(raw_bytes), filename=tmp_name, metadata=metadata)
        tmp_gf  = fs.get(tmp_oid)
        _       = tmp_gf.read(4)
        tmp_gf.close()
        for old in fs.find({"filename": _PARQUET_FNAME}):
            fs.delete(old._id)
        new_oid = fs.put(io.BytesIO(raw_bytes), filename=_PARQUET_FNAME, metadata=metadata)
        fs.delete(tmp_oid)
        tmp_oid = None
        _log(f"[MongoDB] Saved {len(df)} rows, {len(df.columns)} columns. "
             f"_id={new_oid}  last_date={df['Date'].max()}")
    except Exception as e:
        if tmp_oid is not None:
            try:
                fs.delete(tmp_oid)
            except Exception:
                pass
        _log_error("save_parquet_to_mongo", e)
        raise
    finally:
        client.close()


# ── Lot size ──────────────────────────────────────────────────────────────────

def get_historical_lot_size(symbol: str, trade_date) -> int:
    if pd.isnull(trade_date):
        return 0
    try:
        dt = pd.to_datetime(trade_date).date()
    except Exception:
        return 0
    if symbol == "NIFTY":
        if dt < date(2024, 5,  1): return 50
        if dt < date(2024, 12, 1): return 25
        if dt < date(2026, 1,  1): return 75
        return 65
    if symbol == "BANKNIFTY":
        if dt < date(2020, 1,  1): return 20
        if dt < date(2023, 7,  1): return 25
        if dt < date(2025, 1,  1): return 15
        if dt < date(2025, 7, 31): return 30
        if dt < date(2026, 1,  1): return 35
        return 30
    return 0


# ── Fyers ─────────────────────────────────────────────────────────────────────

def _build_fyers() -> fyersModel.FyersModel:
    os.makedirs("/tmp/fyers_logs", exist_ok=True)
    return fyersModel.FyersModel(
        client_id=_ssm_get("/fyers/APP_ID"),
        token=_ssm_get("/fyers/ACCESS_TOKEN"),
        is_async=False,
        log_path="/tmp/fyers_logs",
    )


def fetch_fyers_5m(fyers, symbol: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    for attempt in range(3):
        try:
            resp = fyers.history(data={
                "symbol":      symbol,
                "resolution":  "5",
                "date_format": "1",
                "range_from":  from_dt.strftime("%Y-%m-%d"),
                "range_to":    to_dt.strftime("%Y-%m-%d"),
                "cont_flag":   "1",
            })
            if resp.get("s") == "ok":
                candles = resp.get("candles", [])
                if not candles:
                    _log(f"[Fyers] OK response but 0 candles for {symbol} on {from_dt}")
                    return pd.DataFrame()
                df = pd.DataFrame(candles, columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                df["Datetime"] = pd.to_datetime(df["Epoch"], unit="s")
                df["Symbol"]   = symbol
                _log(f"[Fyers] {symbol} {from_dt}: {len(df)} bars fetched")
                return df[["Symbol", "Datetime", "Open", "High", "Low", "Close", "Volume"]]
            msg = str(resp.get("message", "")).lower()
            if any(t in msg for t in ["rate limit", "429", "too many"]):
                _log(f"[Fyers] Rate limit on {symbol}, sleeping 10 s (attempt {attempt+1}/3)")
                time.sleep(10)
                continue
            _log(f"[Fyers] Non-ok for {symbol}: status={resp.get('s')} "
                 f"message={resp.get('message')} (attempt {attempt+1}/3)")
            return pd.DataFrame()
        except Exception as e:
            _log_error(f"fetch_fyers_5m symbol={symbol} attempt={attempt+1}", e)
            time.sleep(5)
    _log(f"[Fyers] All 3 attempts exhausted for {symbol} — returning empty DataFrame")
    return pd.DataFrame()


# ── NSE Bhavcopy ──────────────────────────────────────────────────────────────

def _nse_session() -> requests.Session:
    sess  = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update(NSE_HEADERS)
    try:
        sess.get("https://www.nseindia.com", timeout=10, verify=False)
    except Exception:
        pass
    return sess


def _download_bhavcopy(sess: requests.Session, tgt: date, tmp: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        if tgt >= FORMAT_CHANGE_DATE_2024:
            url  = NSE_URL_NEW.format(day=tgt)
            _log(f"[Bhavcopy] Downloading (2024+ format): {url}")
            resp = sess.get(url, timeout=20, verify=False)
            if resp.status_code == 200 and len(resp.content) > 1_000:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
                    if not csv_name:
                        return None, "No CSV in zip"
                    z.extract(csv_name, tmp)
                    return os.path.join(tmp, csv_name), None
            return None, f"HTTP {resp.status_code}, body_len={len(resp.content)}"
        elif tgt >= FORMAT_CHANGE_DATE_2023:
            params = {
                "archives": '[{"name":"F&O - Bhavcopy(csv)","type":"archives","category":"derivatives","section":"equity"}]',
                "date":     tgt.strftime("%d-%b-%Y"),
                "type":     "equity",
                "mode":     "single",
            }
            _log(f"[Bhavcopy] Downloading (2023 API format) for {tgt}")
            resp = sess.get(NSE_API_URL, params=params, timeout=25, verify=False)
            if resp.status_code == 200 and len(resp.content) > 1_000:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
                    if not csv_name:
                        return None, "No CSV in API zip"
                    z.extract(csv_name, tmp)
                    return os.path.join(tmp, csv_name), None
            return None, f"HTTP {resp.status_code}, body_len={len(resp.content)}"
        else:
            url  = NSE_URL_OLD.format(day=tgt)
            _log(f"[Bhavcopy] Downloading (legacy format): {url}")
            resp = sess.get(url, timeout=20, verify=False)
            if resp.status_code == 200 and len(resp.content) > 1_000:
                path = os.path.join(tmp, f"bhav_{tgt.strftime('%d%m%Y')}.csv")
                with open(path, "wb") as fh:
                    fh.write(resp.content)
                return path, None
            return None, f"HTTP {resp.status_code}, body_len={len(resp.content)}"
    except Exception as e:
        _log_error(f"_download_bhavcopy for {tgt}", e)
        return None, f"{e}\n{traceback.format_exc()}"


def _parse_bhavcopy(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        _log(f"[Bhavcopy] Raw CSV: {len(df)} rows, cols={list(df.columns[:10])}")

        if "TckrSymb" in df.columns and "NewBrdLotQty" in df.columns:
            _log("[Bhavcopy] Detected format: 2024+ (NewBrdLotQty present)")
            for alias in ("BizDt", "TradDt", "RptgDt"):
                if alias in df.columns and "BizDt" not in df.columns:
                    df["BizDt"] = df[alias]
            df = df[df["TckrSymb"] == TARGET_OPTION].copy()
            df = df[(df["OptnTp"].isin(["CE", "PE"])) & (df["StrkPric"] > 0)].copy()
            if df.empty:
                _log(f"[Bhavcopy] No rows remain after filtering for {TARGET_OPTION} CE/PE")
                return pd.DataFrame()
            df = df.rename(columns={
                "TckrSymb": "Symbol",  "OptnTp": "Type",       "XpryDt": "Expiry",
                "StrkPric": "Strike",  "BizDt":  "Datetime",   "OpnPric": "Open",
                "HghPric":  "High",    "LwPric":  "Low",       "ClsPric": "Close",
                "TtlTradgVol": "Volume","OpnIntrst": "OI",     "ChngInOpnIntrst": "ChgOI",
                "SttlmPric": "SettlePrice","TtlNbOfTxsExctd": "Trades",
                "NewBrdLotQty": "LotSize",
            })
        elif "TckrSymb" in df.columns and "FinInstrmNm" in df.columns:
            _log("[Bhavcopy] Detected format: 2023 API (FinInstrmNm present)")
            for alias in ("RptgDt", "TradDt", "BizDt"):
                if alias in df.columns:
                    df["RptgDt"] = df[alias]
                    break
            df = df[df["TckrSymb"] == TARGET_OPTION].copy()
            df = df[(df["OptnTp"].isin(["CE", "PE"])) & (df["StrkPric"] > 0)].copy()
            if df.empty:
                _log(f"[Bhavcopy] No rows remain after filtering for {TARGET_OPTION} CE/PE")
                return pd.DataFrame()
            df = df.rename(columns={
                "TckrSymb": "Symbol",  "OptnTp": "Type",       "XpryDt": "Expiry",
                "StrkPric": "Strike",  "RptgDt": "Datetime",   "OpnPric": "Open",
                "HghPric":  "High",    "LwPric":  "Low",       "ClsPric": "Close",
                "TtlTradgVol": "Volume","OpnIntrst": "OI",     "ChngInOpnIntrst": "ChgOI",
                "SttlmPric": "SettlePrice","TtlNbOfTxsExctd": "Trades",
            })
            df["LotSize"] = 0
        elif "INSTRUMENT" in df.columns and "SYMBOL" in df.columns:
            _log("[Bhavcopy] Detected format: legacy (INSTRUMENT/SYMBOL columns)")
            df = df[
                (df["SYMBOL"] == TARGET_OPTION) &
                (df["INSTRUMENT"] == "OPTIDX") &
                (df["OPTION_TYP"].isin(["CE", "PE"]))
            ].copy()
            if df.empty:
                _log(f"[Bhavcopy] No rows remain after filtering for {TARGET_OPTION} CE/PE")
                return pd.DataFrame()
            df = df.rename(columns={
                "SYMBOL": "Symbol",    "OPTION_TYP": "Type",   "EXPIRY_DT": "Expiry",
                "STRIKE_PR": "Strike", "TIMESTAMP":  "Datetime","OPEN": "Open",
                "HIGH": "High",        "LOW": "Low",            "CLOSE": "Close",
                "CONTRACTS": "Volume", "OPEN_INT": "OI",        "CHG_IN_OI": "ChgOI",
                "SETTLE_PR": "SettlePrice",
            })
            df["Trades"]  = 0
            df["LotSize"] = 0
        else:
            _log(f"[Bhavcopy] !! UNKNOWN format — first 8 cols: {list(df.columns[:8])} — "
                 "no parsing attempted")
            return pd.DataFrame()

        for dcol in ("Datetime", "Expiry"):
            if dcol not in df.columns:
                continue
            raw    = df[dcol].astype(str).str.strip()
            parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
            bad    = parsed.isna()
            if bad.any():
                parsed[bad] = pd.to_datetime(raw[bad].str.title(), format="%d-%b-%Y", errors="coerce")
            df[dcol] = parsed.dt.strftime("%Y-%m-%d")
            before = len(df)
            df = df.dropna(subset=[dcol])
            after = len(df)
            if before != after:
                _log(f"[Bhavcopy] Dropped {before-after} rows with unparseable {dcol}")

        for col in ("LotSize", "Trades", "ChgOI"):
            if col not in df.columns:
                df[col] = 0

        def _lot(row):
            v = row.get("LotSize", 0)
            if v and v > 0:
                return v
            return get_historical_lot_size(row.get("Symbol", ""), row.get("Datetime"))

        df["LotSize"] = df.apply(_lot, axis=1)

        for c in ["Strike", "Open", "High", "Low", "Close", "Volume",
                  "OI", "ChgOI", "SettlePrice", "Trades", "LotSize"]:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

        canonical = ["Symbol", "Type", "Expiry", "Strike", "Datetime",
                     "Open", "High", "Low", "Close", "Volume", "OI",
                     "SettlePrice", "Trades", "LotSize"]
        for c in canonical:
            if c not in df.columns:
                df[c] = 0

        result = df[canonical].reset_index(drop=True)
        _log(f"[Bhavcopy] Parsed {len(result)} option rows "
             f"(CE={len(result[result['Type']=='CE'])}, "
             f"PE={len(result[result['Type']=='PE'])})")
        return result

    except Exception as e:
        _log_error("_parse_bhavcopy", e)
        return pd.DataFrame()


# ── IV & Greeks ───────────────────────────────────────────────────────────────

def _ndtr(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    _erf = np.frompyfunc(erf, 1, 1)
    return 0.5 * (1.0 + _erf(x / sqrt(2.0)).astype(float))


def _bs_price(S, K, T, r, sigma, flag):
    T    = np.maximum(T, 1e-6)
    sig  = np.maximum(sigma, 1e-6)
    d1   = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2   = d1 - sig * np.sqrt(T)
    disc = np.exp(-r * T)
    call = S * _ndtr(d1) - K * disc * _ndtr(d2)
    put  = K * disc * _ndtr(-d2) - S * _ndtr(-d1)
    return np.where(np.asarray(flag) == "c", call, put)


def _bisect_iv(price, S, K, T, r, flag, tol=1e-5, max_iter=60):
    lo  = np.full_like(price, 1e-4, dtype=float)
    hi  = np.full_like(price, 6.0,  dtype=float)
    mid = np.empty_like(price, dtype=float)
    for _ in range(max_iter):
        mid   = 0.5 * (lo + hi)
        p_mid = _bs_price(S, K, T, r, mid, flag)
        lo    = np.where(p_mid < price, mid, lo)
        hi    = np.where(p_mid > price, mid, hi)
        if np.max(hi - lo) < tol:
            break
    intrinsic = np.where(
        np.asarray(flag) == "c",
        np.maximum(S - K, 0.0),
        np.maximum(K - S, 0.0),
    )
    mid[((price <= intrinsic + 1e-6) | (mid <= 1e-4) | (mid >= 5.99))] = np.nan
    return mid


def _compute_delta_np(S, K, T, r, sigma, flag):
    T     = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1    = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    cd    = _ndtr(d1)
    return np.where(np.asarray(flag) == "c", cd, cd - 1.0)


def _compute_gamma_np(S, K, T, r, sigma):
    T     = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1    = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pdf   = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * np.sqrt(T))


def compute_iv_and_greeks(opts: pd.DataFrame, spot: float, india_vix: float) -> pd.DataFrame:
    opts            = opts.copy()
    opts["SpotPrice"] = spot
    opts["IndiaVIX"]  = india_vix
    fallback_iv       = max(india_vix / 100.0, 0.20)

    opts["Expiry_dt"]   = pd.to_datetime(opts["Expiry"])
    opts["Datetime_dt"] = pd.to_datetime(opts["Datetime"])
    opts["TimeYear"]    = ((opts["Expiry_dt"] - opts["Datetime_dt"]).dt.days / 365.0).clip(lower=1e-4)
    opts["flag"]        = opts["Type"].map({
        "CE": "c", "PE": "p", "ce": "c", "pe": "p",
        "C":  "c", "P":  "p", "c":  "c", "p":  "p",
    })
    before = len(opts)
    opts = opts.dropna(subset=["flag"]).reset_index(drop=True)
    if before != len(opts):
        _log(f"[IV] Dropped {before-len(opts)} rows with unrecognised option type")
    if opts.empty:
        _log("[IV] No rows after flag mapping — returning empty")
        return opts

    opts["Moneyness"] = np.log(opts["Strike"] / opts["SpotPrice"])

    try:
        from py_vollib_vectorized import vectorized_implied_volatility
        raw_iv = vectorized_implied_volatility(
            opts["Close"].values, opts["SpotPrice"].values, opts["Strike"].values,
            opts["TimeYear"].values, RISK_FREE_RATE, opts["flag"].values,
            return_as="numpy", on_error="ignore",
        )
        _log("[IV] py_vollib_vectorized used for IV calculation")
    except Exception as e:
        _log(f"[IV] py_vollib unavailable ({e}) — using bisection fallback")
        raw_iv = _bisect_iv(
            opts["Close"].values, opts["SpotPrice"].values, opts["Strike"].values,
            opts["TimeYear"].values, RISK_FREE_RATE, opts["flag"].values,
        )
    opts["Raw_IV"] = raw_iv

    nan_iv_count = opts["Raw_IV"].isna().sum()
    _log(f"[IV] Raw IV: {len(opts)-nan_iv_count} valid, {nan_iv_count} NaN "
         f"out of {len(opts)} rows")

    def _fit_smile(grp: pd.DataFrame) -> pd.Series:
        sp    = grp["SpotPrice"].iloc[0]
        valid = (grp["Raw_IV"] > 0.01) & (grp["Raw_IV"] < 2.0) & grp["Raw_IV"].notna()
        is_c  = grp["flag"] == "c"
        is_p  = grp["flag"] == "p"
        otm   = ((is_c & (grp["Strike"] >= sp * 0.95)) | (is_p & (grp["Strike"] <= sp * 1.05)))
        anch  = grp.loc[valid & otm]
        if len(anch) < 4:
            anch = grp.loc[valid]
        if len(anch) < 3:
            fb = grp["IndiaVIX"].iloc[0] / 100.0
            return pd.Series(max(fb, 0.2), index=grp.index)
        try:
            x, y   = anch["Moneyness"].values, anch["Raw_IV"].values
            w      = 1.0 / (np.abs(x) + 0.1)
            coeffs = np.polyfit(x, y, 2, w=w)
            fitted = np.clip(np.polyval(coeffs, grp["Moneyness"].values), 0.05, 3.0)
            return pd.Series(fitted, index=grp.index)
        except Exception:
            fb = grp["IndiaVIX"].iloc[0] / 100.0
            return pd.Series(max(fb, 0.2), index=grp.index)

    opts["GroupID"] = opts["Datetime"].astype(str) + "_" + opts["Expiry"].astype(str)
    try:
        clean_iv = opts.groupby("GroupID", group_keys=False).apply(
            _fit_smile, include_groups=False
        )
    except TypeError:
        clean_iv = opts.groupby("GroupID", group_keys=False).apply(_fit_smile)
    opts["IV"] = clean_iv.fillna(fallback_iv).values

    try:
        from py_vollib_vectorized import get_all_greeks
        gdf = get_all_greeks(
            opts["flag"].values, opts["SpotPrice"].values, opts["Strike"].values,
            opts["TimeYear"].values, RISK_FREE_RATE, opts["IV"].values,
            model="black_scholes", return_as="dataframe",
        ).reset_index(drop=True)
        opts["delta"] = gdf.get("delta", np.nan)
        opts["gamma"] = gdf.get("gamma", np.nan)
        _log("[IV] Greeks computed via py_vollib_vectorized")
    except Exception as e:
        _log(f"[IV] py_vollib Greeks unavailable ({e}) — using numpy fallback")
        opts["delta"] = _compute_delta_np(
            opts["SpotPrice"].values, opts["Strike"].values,
            opts["TimeYear"].values, RISK_FREE_RATE, opts["IV"].values, opts["flag"].values,
        )
        opts["gamma"] = _compute_gamma_np(
            opts["SpotPrice"].values, opts["Strike"].values,
            opts["TimeYear"].values, RISK_FREE_RATE, opts["IV"].values,
        )
    return opts


# ── Dukascopy ─────────────────────────────────────────────────────────────────

# ── Dukascopy instrument lookup (built once at import time) ───────────────────
def _build_duka_instrument_map() -> dict:
    """
    Scan dukascopy_python.instruments and return a dict mapping
    uppercase symbol-id (e.g. "USA500IDXUSD") → instrument constant object.
    """
    try:
        import dukascopy_python.instruments as _m
        mapping = {}
        for attr in dir(_m):
            if not attr.startswith("INSTRUMENT_"):
                continue
            val = getattr(_m, attr)
            if hasattr(val, "id"):
                mapping[val.id.upper()] = val
        _log(f"[Dukascopy] Instrument map built: {len(mapping)} entries")
        return mapping
    except Exception as e:
        _log_error("_build_duka_instrument_map", e)
        return {}

_DUKA_INSTRUMENT_MAP: dict = _build_duka_instrument_map()


def fetch_dukascopy_day(instrument: str, tgt: date) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV candles for *instrument* on *tgt* using dukascopy-python.

    Returns a DataFrame with lowercase columns:
        timestamp (datetime, IST, tz-naive), symbol, open, high, low, close, volume
    """
    import dukascopy_python

    inst_key = instrument.upper()
    instrument_obj = _DUKA_INSTRUMENT_MAP.get(inst_key)
    if instrument_obj is None:
        _log(f"[Dukascopy] !! No instrument constant found for '{instrument}' "
             f"— available keys sample: {list(_DUKA_INSTRUMENT_MAP.keys())[:8]}")
        return pd.DataFrame()

    start_dt = datetime(tgt.year, tgt.month, tgt.day)
    end_dt   = start_dt + timedelta(days=1)   # end is exclusive in the API

    try:
        df = dukascopy_python.fetch(
            instrument=instrument_obj,
            interval=dukascopy_python.INTERVAL_MIN_5,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start_dt,
            end=end_dt,
        )
    except Exception as e:
        _log_error(f"fetch_dukascopy_day instrument={instrument} date={tgt}", e)
        return pd.DataFrame()

    if df is None or df.empty:
        _log(f"[Dukascopy] {instrument} {tgt}: 0 records returned by dukascopy-python")
        return pd.DataFrame()

    # ── Normalise: move DatetimeIndex → column named 'timestamp' ─────────────
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    # The index column may be named 'timestamp', 'time', 'date', or 'index'
    time_col = next(
        (c for c in df.columns if c in ("timestamp", "time", "date", "index")), None
    )
    if time_col is None:
        _log(f"[Dukascopy] {instrument} {tgt}: no time column found — got: {list(df.columns)}")
        return pd.DataFrame()
    if time_col != "timestamp":
        df.rename(columns={time_col: "timestamp"}, inplace=True)

    # ── UTC → IST ──────────────────────────────────────────────────────────────
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        ts = ts + IST_OFFSET
    df["timestamp"] = ts

    # ── Keep only the target IST date, weekdays only ──────────────────────────
    df = df[df["timestamp"].dt.dayofweek < 5].copy()
    df = df[df["timestamp"].dt.date == tgt].reset_index(drop=True)

    if df.empty:
        _log(f"[Dukascopy] {instrument} {tgt}: 0 bars after IST date-filter")
        return pd.DataFrame()

    # ── Flatline guard ────────────────────────────────────────────────────────
    if df["high"].max() == df["low"].min():
        _log(f"[Dukascopy] {instrument} {tgt}: flat data (high==low) — discarding")
        return pd.DataFrame()

    df["symbol"] = instrument.lower()
    _log(f"[Dukascopy] {instrument} {tgt}: {len(df)} bars fetched via dukascopy-python")

    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ── Daily feature row ─────────────────────────────────────────────────────────

def _daily_row(tgt, spot_5m, vix_5m, opts, global_5m, prev_gc) -> dict:
    row: dict = {"Date": tgt}

    if not spot_5m.empty:
        s  = spot_5m.sort_values("Datetime").reset_index(drop=True)
        lr = np.log(s["Close"] / s["Close"].shift(1))
        ac = (s["Close"] - s["Close"].shift(1)).abs()
        row["Spot_Open"]        = float(s["Open"].iloc[0])
        row["Spot_High"]        = float(s["High"].max())
        row["Spot_Low"]         = float(s["Low"].min())
        row["Spot_Close"]       = float(s["Close"].iloc[-1])
        row["Spot_Volume"]      = float(s["Volume"].sum())
        row["Realized_Vol_Ann"] = float(lr.std() * math.sqrt(75 * 252))
        row["Path_Length"]      = float(ac.sum())
        span = abs(row["Spot_Close"] - row["Spot_Open"])
        row["Trend_Strength"]   = span / max(row["Path_Length"], 1e-9)
    else:
        for k in ("Spot_Open","Spot_High","Spot_Low","Spot_Close","Spot_Volume",
                  "Realized_Vol_Ann","Path_Length","Trend_Strength"):
            row[k] = float("nan")

    row["VIX_Close"] = (
        float(vix_5m.sort_values("Datetime")["Close"].iloc[-1])
        if not vix_5m.empty else float("nan")
    )

    if not opts.empty and "IV" in opts.columns and "delta" in opts.columns:
        is_c = opts["Type"].isin(["C", "CE"])
        is_p = opts["Type"].isin(["P", "PE"])
        row["Call_Volume"] = float(opts.loc[is_c, "Volume"].sum())
        row["Put_Volume"]  = float(opts.loc[is_p, "Volume"].sum())
        row["Call_OI"]     = float(opts.loc[is_c, "OI"].sum())
        row["Put_OI"]      = float(opts.loc[is_p, "OI"].sum())
        row["PCR_OI"]      = row["Put_OI"]    / max(row["Call_OI"],    1.0)
        row["PCR_Volume"]  = row["Put_Volume"] / max(row["Call_Volume"], 1.0)

        vld = opts.dropna(subset=["delta", "IV"])
        def _25d(sub):
            if sub.empty:
                return float("nan")
            return float(sub.assign(dd=(sub["delta"].abs() - 0.25).abs()).nsmallest(1, "dd")["IV"].iloc[0])

        p25 = _25d(vld[is_p.reindex(vld.index, fill_value=False)])
        c25 = _25d(vld[is_c.reindex(vld.index, fill_value=False)])
        row["Put_IV_25d"]       = p25
        row["Call_IV_25d"]      = c25
        row["Options_Skew_25d"] = (
            p25 - c25 if not (math.isnan(p25) or math.isnan(c25)) else float("nan")
        )

        sp_   = row.get("Spot_Close") or 0.0
        gex_r = opts.dropna(subset=["gamma", "OI"])
        if not gex_r.empty and sp_ > 0:
            s_g = np.where(
                is_c.reindex(gex_r.index, fill_value=False),
                gex_r["gamma"].values, -gex_r["gamma"].values,
            )
            row["GEX_Raw"] = float(
                (s_g * gex_r["OI"].values * gex_r["LotSize"].values * sp_**2 / 100.0).sum()
            )
        else:
            row["GEX_Raw"] = float("nan")
    else:
        missing_reason = "opts empty" if opts.empty else "IV/delta cols missing"
        _log(f"[DailyRow] {tgt}: options features set to NaN — {missing_reason}")
        for k in ("Call_Volume","Put_Volume","Call_OI","Put_OI","PCR_OI",
                  "PCR_Volume","Put_IV_25d","Call_IV_25d","Options_Skew_25d","GEX_Raw"):
            row[k] = float("nan")

    row["GEX_Zscore"] = float("nan")
    row["GEX_Pos"]    = float("nan")
    row["GEX_Neg"]    = float("nan")

    for sym_name, df_g in global_5m.items():
        if df_g.empty:
            _log(f"[DailyRow] {tgt}: {sym_name} global data empty — Drift/Momentum set to NaN")
            row[f"Intraday_Drift_{sym_name}"]     = float("nan")
            row[f"Overnight_Momentum_{sym_name}"] = float("nan")
            continue
        df_g    = df_g.copy()
        df_g["_t"] = df_g["timestamp"].dt.time
        t0900   = pd.Timestamp("09:00").time()
        t0915   = pd.Timestamp("09:15").time()
        t1530   = pd.Timestamp("15:30").time()
        sess    = df_g[(df_g["_t"] >= t0900) & (df_g["_t"] <= t1530)].sort_values("timestamp")
        if sess.empty:
            _log(f"[DailyRow] {tgt}: {sym_name} — no bars in 09:00-15:30 IST window")
            row[f"Intraday_Drift_{sym_name}"]     = float("nan")
            row[f"Overnight_Momentum_{sym_name}"] = float("nan")
            continue
        open_0900  = float(sess.iloc[0]["open"])
        after_0915 = sess[sess["_t"] >= t0915]
        open_0915  = float(after_0915.iloc[0]["open"]) if not after_0915.empty else open_0900
        close_1530 = float(sess.iloc[-1]["close"])
        row[f"Intraday_Drift_{sym_name}"]     = math.log(close_1530 / open_0915) if open_0915 > 0 else float("nan")
        prev_c = prev_gc.get(sym_name)
        row[f"Overnight_Momentum_{sym_name}"] = (
            math.log(open_0900 / prev_c) if (prev_c and prev_c > 0 and open_0900 > 0) else float("nan")
        )
        row[f"{_GC_PFX}{sym_name}"] = close_1530

    row["Target_Ret_1D"]   = float("nan")
    row["Target_Ret_5D"]   = float("nan")
    row["Trailing_Ret_5D"] = float("nan")
    row["VRP_Daily"]       = (
        (row.get("VIX_Close") or float("nan")) / 100.0
        - (row.get("Realized_Vol_Ann") or float("nan"))
    )
    return row


# ── Rolling features ──────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    gm = df["GEX_Raw"].rolling(20, min_periods=5).mean()
    gs = df["GEX_Raw"].rolling(20, min_periods=5).std()
    df["GEX_Zscore"] = ((df["GEX_Raw"] - gm) / (gs + 1e-6)).clip(-3.0, 3.0)
    df["GEX_Pos"]    = df["GEX_Zscore"].clip(lower=0.0)
    df["GEX_Neg"]    = (-df["GEX_Zscore"]).clip(lower=0.0)

    sc = df["Spot_Close"]
    df["Target_Ret_1D"]   = np.log(sc.shift(-1) / sc)
    df["Target_Ret_5D"]   = np.log(sc.shift(-5) / sc)
    df["Trailing_Ret_5D"] = np.log(sc / sc.shift(5))
    df["VRP_Daily"]       = df["VIX_Close"] / 100.0 - df["Realized_Vol_Ann"]

    _v = df["Realized_Vol_Ann"]
    df["Vol_Change_1D"]       = _v.pct_change(1).clip(-2, 2)
    df["Vol_Change_5D"]       = _v.pct_change(5).clip(-2, 2)
    df["Vol_of_Vol_10D"]      = _v.rolling(10).std()
    df["VoV_5D"]              = _v.rolling(5).std()
    df["Vol_Regime_Zscore"]   = ((_v - _v.rolling(20).mean()) / (_v.rolling(20).std() + 1e-6)).clip(-3, 3)
    df["Recent_Vol_Drawdown"] = (_v / (_v.rolling(10).max() + 1e-6)).clip(0.05, 1.5)
    df["Vol_MA_Ratio"]        = (_v.rolling(5).mean() / (_v.rolling(20).mean() + 1e-6)).clip(0.1, 5.0)
    df["Vol_Accel_3D"]        = (_v / (_v.shift(3) + 1e-6)).clip(0.2, 5.0)

    below_mean = (_v < _v.rolling(20, min_periods=5).mean()).astype(int)
    streak, s  = [], 0
    for v in below_mean:
        s = s + 1 if v else 0
        streak.append(s)
    df["Vol_Compression_Streak"] = np.log1p(np.array(streak, dtype=np.float32))

    _c, _o, _h, _l = df["Spot_Close"], df["Spot_Open"], df["Spot_High"], df["Spot_Low"]
    _lr = np.log(_c / _c.shift(1))
    _hl = np.log(_h / _l).clip(0, 0.10)
    df["HL_Range_Zscore"]   = ((_hl - _hl.rolling(20, min_periods=5).mean()) /
                                (_hl.rolling(20, min_periods=5).std() + 1e-6)).clip(-3, 3)
    df["Overnight_Gap_Abs"] = np.log(_o / _c.shift(1)).abs().clip(0, 0.05)
    _upper = np.log(_h / np.maximum(_o, _c)).clip(0, 0.05)
    _lower = np.log(np.minimum(_o, _c) / _l).clip(0, 0.05)
    df["Shadow_Imbalance"]  = (_lower - _upper).clip(-0.05, 0.05)

    _dn = _lr.where(_lr < 0, 0.0)
    _up = _lr.where(_lr > 0, 0.0)
    df["SemiVar_Down_10D"]  = (_dn.rolling(10, min_periods=5).std() * math.sqrt(252)).clip(0, 1.5).fillna(0)
    df["SemiVar_Up_10D"]    = (_up.rolling(10, min_periods=5).std() * math.sqrt(252)).clip(0, 1.5).fillna(0)
    df["SemiVar_Ratio_10D"] = (df["SemiVar_Down_10D"] / (df["SemiVar_Up_10D"] + 1e-6)).clip(0.1, 5.0)

    df["NIFTY_Overnight_Gap"] = np.log(_o / _c.shift(1)).clip(-0.05, 0.05)
    df["NIFTY_Intraday_Ret"]  = np.log(_c / _o).clip(-0.05, 0.05)
    df["Recent_Max_AbsRet"]   = df["Target_Ret_1D"].shift(1).abs().rolling(5).max()

    df["PCR_OI_Change_1D"] = df["PCR_OI"].pct_change(1).clip(-2, 2)
    df["PCR_OI_Zscore"]    = (
        (df["PCR_OI"] - df["PCR_OI"].rolling(20, min_periods=5).mean()) /
        (df["PCR_OI"].rolling(20, min_periods=5).std() + 1e-6)
    ).clip(-3, 3)

    for col in ("Overnight_Momentum_sp500", "Overnight_Momentum_crudeoil", "Overnight_Momentum_hang_seng"):
        if col in df.columns:
            sg = np.sign(df[col])
            df[f"{col}_Consist_3D"] = sg.rolling(3).sum()
            df[f"{col}_Consist_5D"] = sg.rolling(5).sum()
        else:
            _log(f"[Rolling] Column {col} missing — Consist_3D/5D set to 0")
            df[f"{col}_Consist_3D"] = 0.0
            df[f"{col}_Consist_5D"] = 0.0

    _is = np.sign(df["NIFTY_Intraday_Ret"])
    df["NIFTY_Intraday_Consist_3D"] = _is.rolling(3).sum()
    df["NIFTY_Intraday_Consist_5D"] = _is.rolling(5).sum()

    _dow = pd.to_datetime(df["Date"]).dt.dayofweek
    df["DTE_R0_Expiry"]  = (_dow == 3).astype(float)
    df["DTE_R1_PreExp"]  = (_dow == 2).astype(float)
    df["DTE_R2_MidWeek"] = (_dow == 1).astype(float)
    df["DTE_R3_PostExp"] = (_dow == 4).astype(float)

    _sp500_mom = df.get("Overnight_Momentum_sp500", pd.Series(0.0, index=df.index))
    df["DTE_PostExp_x_SP500Mom"] = df["DTE_R3_PostExp"] * _sp500_mom
    df["GEX_x_DTE_R0"]          = df["GEX_Zscore"]     * df["DTE_R0_Expiry"]

    df.fillna(0, inplace=True)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_row(row: dict, existing: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    tgt = row.get("Date")
    if isinstance(tgt, date) and tgt.weekday() >= 5:
        return False, f"{tgt} is a weekend"
    sp = row.get("Spot_Close", float("nan"))
    if math.isnan(sp) or sp <= 0:
        return False, "Spot_Close missing/zero — NSE likely closed"
    if not (_NIFTY_LO < sp < _NIFTY_HI):
        return False, f"Spot_Close={sp:.1f} outside plausible range [{_NIFTY_LO}, {_NIFTY_HI}]"
    rv = row.get("Realized_Vol_Ann", float("nan"))
    if not math.isnan(rv) and (rv <= 0 or rv > 3.0):
        return False, f"Realized_Vol_Ann={rv:.4f} implausible (must be 0 < rv <= 3.0)"
    if existing is not None and not existing.empty:
        if tgt in set(pd.to_datetime(existing["Date"]).dt.date):
            return False, f"{tgt} already in master parquet (duplicate)"
    return True, "ok"


# ── Column audit ──────────────────────────────────────────────────────────────

def _audit_columns(
    existing_df: pd.DataFrame,
    new_rows: list,
    combined: pd.DataFrame,
) -> None:
    _log_section("COLUMN AUDIT")

    existing_pub = sorted(c for c in existing_df.columns if not c.startswith(_GC_PFX))
    new_row_keys = sorted(
        k for k in (new_rows[0].keys() if new_rows else []) if not k.startswith(_GC_PFX)
    )
    combined_pub = sorted(c for c in combined.columns if not c.startswith(_GC_PFX))

    _log(f"Columns in existing parquet  : {len(existing_pub)}")
    _log(f"Keys produced by _daily_row  : {len(new_row_keys)}")
    _log(f"Columns in combined parquet  : {len(combined_pub)}")

    missing_in_combined = sorted(set(existing_pub) - set(combined_pub))
    if missing_in_combined:
        _log(f"\n!! {len(missing_in_combined)} column(s) present in existing parquet "
             "but MISSING from combined result — data loss risk:")
        for c in missing_in_combined:
            _log(f"   MISSING  → {c}")
    else:
        _log("\n✓ All existing columns are present in the combined parquet")

    not_computed_in_new = sorted(set(existing_pub) - set(new_row_keys))
    if not_computed_in_new:
        _log(f"\n⚠ {len(not_computed_in_new)} column(s) exist in the old data but were NOT "
             "directly emitted by _daily_row/_compute_rolling in this run "
             "(carried over from existing rows only):")
        for c in not_computed_in_new:
            _log(f"   NOT-RECOMPUTED → {c}")
    else:
        _log("\n✓ _daily_row produced all columns that exist in the existing parquet")

    new_cols = sorted(set(combined_pub) - set(existing_pub))
    if new_cols:
        _log(f"\n+ {len(new_cols)} NEW column(s) added this run:")
        for c in new_cols:
            _log(f"   NEW → {c}")

    if new_rows:
        new_df_audit = pd.DataFrame(new_rows)
        pub_cols_in_new = [c for c in new_df_audit.columns if not c.startswith(_GC_PFX)]
        nan_summary = (
            new_df_audit[pub_cols_in_new]
            .isna()
            .sum()
            .loc[lambda s: s > 0]
            .sort_values(ascending=False)
        )
        if not nan_summary.empty:
            _log(f"\nNaN counts across {len(new_rows)} new row(s) "
                 "(columns with at least 1 NaN before rolling fill):")
            for col, cnt in nan_summary.items():
                _log(f"   NaN × {cnt:>3}  → {col}")
        else:
            _log("\n✓ No NaN values in newly computed rows (before rolling fill)")

    _log_section("END COLUMN AUDIT")


# ── Handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        _log_section("Nifty Feature Pipeline — START")

        tmp_bhav = "/tmp/bhav_dl"
        os.makedirs(tmp_bhav,          exist_ok=True)
        os.makedirs("/tmp/fyers_logs", exist_ok=True)

        # ── Load existing parquet ──────────────────────────────────────────────
        existing_df = load_parquet_from_mongo()
        if existing_df is not None and not existing_df.empty:
            last_date = pd.to_datetime(existing_df["Date"]).dt.date.max()
            _log(f"Existing parquet: {len(existing_df)} rows, "
                 f"{len(existing_df.columns)} total columns, last={last_date}")
        else:
            existing_df = pd.DataFrame()
            last_date   = None
            _log("No existing parquet — fresh start")

        # ── Determine date range ───────────────────────────────────────────────
        # CHANGE 1: Always reprocess last 2 trading days so that global market
        # Overnight_Momentum features (which depend on the previous day's close)
        # are correctly recalculated with fresh prev_gc values.
        now_ist   = datetime.utcnow() + IST_OFFSET
        today_ist = now_ist.date()
        max_fetch = int(os.environ.get("MAX_FETCH_DAYS", "5"))

        if last_date:
            # Walk back to find the start of the 2-trading-day reprocess window
            reprocess_start = last_date
            trading_days_found = 0
            candidate = last_date
            while trading_days_found < 2:
                if candidate.weekday() < 5:
                    trading_days_found += 1
                    reprocess_start = candidate
                if trading_days_found < 2:
                    candidate -= timedelta(days=1)
            start = reprocess_start
            _log(f"[DateRange] Reprocessing from {start} (last 2 trading days) to pick up "
                 f"correct global prev_gc carry-forward")
        else:
            lookback = int(os.environ.get("START_LOOKBACK_DAYS", "3"))
            start    = today_ist - timedelta(days=lookback)

        dates_to_fetch = []
        d = start
        while d <= today_ist and len(dates_to_fetch) < max_fetch:
            if d.weekday() < 5:
                dates_to_fetch.append(d)
            d += timedelta(days=1)

        if not dates_to_fetch:
            _log("Parquet already up-to-date — nothing to do")
            return {"statusCode": 200, "body": "already_up_to_date"}

        _log(f"Dates queued for processing ({len(dates_to_fetch)}): {dates_to_fetch}")

        # ── Init clients ───────────────────────────────────────────────────────
        fyers    = _build_fyers()
        nse_sess = _nse_session()

        # Seed prev_gc from the day BEFORE our reprocess window start
        prev_gc: Dict[str, Optional[float]] = {s: None for s in DUKA_SYMBOLS}
        if not existing_df.empty:
            for sym in DUKA_SYMBOLS:
                col = f"{_GC_PFX}{sym}"
                if col in existing_df.columns:
                    ser = existing_df.loc[
                        pd.to_datetime(existing_df["Date"]).dt.date < start, col
                    ].dropna()
                    if len(ser):
                        prev_gc[sym] = float(ser.iloc[-1])
            _log(f"Seeded prev_gc from existing data: "
                 f"{ {k: f'{v:.4f}' if v else 'None' for k, v in prev_gc.items()} }")

        # Strip dates being reprocessed from existing_df so _validate_row
        # does not reject them as duplicates — drop_duplicates(keep="last")
        # in the merge step will keep the freshly computed versions.
        reprocess_dates = set(dates_to_fetch)
        if not existing_df.empty and reprocess_dates:
            existing_df_for_validation = existing_df[
                ~pd.to_datetime(existing_df["Date"]).dt.date.isin(reprocess_dates)
            ].reset_index(drop=True)
            _log(f"[Reprocess] Stripped {len(reprocess_dates)} date(s) from duplicate-check "
                 f"view: {sorted(reprocess_dates)}")
        else:
            existing_df_for_validation = existing_df

        skipped_dates = []
        new_rows: list = []

        # ── Per-date loop ──────────────────────────────────────────────────────
        for tgt in dates_to_fetch:
            _log(f"\n{'─'*40}")
            _log(f"Processing: {tgt}")
            _log(f"{'─'*40}")

            spot_5m = fetch_fyers_5m(fyers, SPOT_SYMBOL, tgt, tgt)
            vix_5m  = fetch_fyers_5m(fyers, VIX_SYMBOL,  tgt, tgt)

            _fyers_raw_count = len(spot_5m)
            if not spot_5m.empty:
                spot_5m = spot_5m[spot_5m["Datetime"].dt.date == tgt].reset_index(drop=True)
                _log(f"[{tgt}] Spot bars after date-filter: {len(spot_5m)} "
                    f"(raw from Fyers: {_fyers_raw_count})")
            if not vix_5m.empty:
                vix_5m  = vix_5m[vix_5m["Datetime"].dt.date  == tgt].reset_index(drop=True)
                _log(f"[{tgt}] VIX bars after date-filter: {len(vix_5m)}")

            if spot_5m.empty:
                _skip_reason = (
                    "Fyers returned 0 candles — token may be expired or market was closed"
                    if _fyers_raw_count == 0
                    else f"date-filter removed all {_fyers_raw_count} Fyers bars (timezone mismatch?)"
                )
                _log(f"[{tgt}] !! {_skip_reason} — skipping date")
                skipped_dates.append(str(tgt))
                for sym_name, sym_code in DUKA_SYMBOLS.items():
                    df_g = fetch_dukascopy_day(sym_code, tgt)
                    if not df_g.empty:
                        sess_end = df_g[
                            df_g["timestamp"].dt.time <= pd.Timestamp("15:30").time()
                        ].sort_values("timestamp")
                        if not sess_end.empty:
                            prev_gc[sym_name] = float(sess_end.iloc[-1]["close"])
                continue

            spot_close = float(spot_5m["Close"].iloc[-1])
            india_vix  = float(vix_5m["Close"].iloc[-1]) if not vix_5m.empty else 15.0
            _log(f"[{tgt}] spot_close={spot_close:.2f}  india_vix={india_vix:.2f}")

            # Bhavcopy
            time.sleep(random.uniform(0.6, 1.4))
            bhav_path, bhav_err = _download_bhavcopy(nse_sess, tgt, tmp_bhav)
            opts = pd.DataFrame()
            if bhav_path:
                raw_opts = _parse_bhavcopy(bhav_path)
                try:
                    os.remove(bhav_path)
                except OSError:
                    pass
                if not raw_opts.empty:
                    opts = compute_iv_and_greeks(raw_opts, spot_close, india_vix)
                    _log(f"[{tgt}] Options after IV/Greeks: {len(opts)} rows, "
                         f"cols={[c for c in opts.columns if c not in raw_opts.columns]}")
                    del raw_opts
                else:
                    _log(f"[{tgt}] !! Bhavcopy parsed to 0 rows — options features will be NaN")
            else:
                _log(f"[{tgt}] !! Bhavcopy unavailable — {bhav_err}")

            # Global markets (Dukascopy)
            global_5m:  Dict[str, pd.DataFrame]   = {}
            current_gc: Dict[str, Optional[float]] = dict(prev_gc)
            for sym_name, sym_code in DUKA_SYMBOLS.items():
                df_g = fetch_dukascopy_day(sym_code, tgt)
                global_5m[sym_name] = df_g
                if not df_g.empty:
                    sess_end = df_g[
                        df_g["timestamp"].dt.time <= pd.Timestamp("15:30").time()
                    ].sort_values("timestamp")
                    if not sess_end.empty:
                        current_gc[sym_name] = float(sess_end.iloc[-1]["close"])

            # Build daily row
            row = _daily_row(tgt, spot_5m, vix_5m, opts, global_5m, prev_gc)

            # Validate against the view that excludes reprocess dates
            ok, reason = _validate_row(
                row, existing_df_for_validation if not existing_df_for_validation.empty else None
            )
            if not ok:
                _log(f"[{tgt}] !! Validation FAILED — {reason} — skipping row")
                skipped_dates.append(str(tgt))
                prev_gc = current_gc
                del spot_5m, vix_5m, opts, global_5m
                gc.collect()
                continue

            new_rows.append(row)
            prev_gc = current_gc

            _log(
                f"[{tgt}] ✓ Row accepted | "
                f"Close={spot_close:.1f} | "
                f"RV={row['Realized_Vol_Ann']:.4f} | "
                f"VIX={row.get('VIX_Close', float('nan')):.2f} | "
                f"PCR_OI={row.get('PCR_OI', 0.0):.3f} | "
                f"GEX_Raw={row.get('GEX_Raw', 0.0):.3e} | "
                f"PCR_Vol={row.get('PCR_Volume', 0.0):.3f} | "
                f"Skew_25d={row.get('Options_Skew_25d', float('nan')):.4f}"
            )
            del spot_5m, vix_5m, opts, global_5m
            gc.collect()

        import shutil
        shutil.rmtree(tmp_bhav, ignore_errors=True)

        if not new_rows:
            _log("\n!! No valid new rows after validation pass")
            _all_skipped = set(str(d) for d in dates_to_fetch) == set(skipped_dates)
            if _all_skipped:
                _log("   !! ALL dates were skipped — most likely cause: Fyers token expired. "
                    "Check the skip-reason lines above.")
            else:
                _log(f"   Skipped dates ({len(skipped_dates)}): {skipped_dates}")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "status":            "no_new_data",
                    "dates_attempted":   [str(d) for d in dates_to_fetch],
                    "dates_skipped":     skipped_dates,
                    "all_dates_skipped": _all_skipped,
                }),
            }
        # ── Merge & rolling features ───────────────────────────────────────────
        _log_section("Merging & computing rolling features")

        new_df         = pd.DataFrame(new_rows)
        new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.date

        _log(f"New rows DataFrame: {len(new_df)} rows × {len(new_df.columns)} columns")
        _log(f"New row columns ({len(new_df.columns)}): "
             f"{sorted(c for c in new_df.columns if not c.startswith(_GC_PFX))}")

        # Merge: existing_df (full, unstripped) + new_df; keep="last" ensures
        # reprocessed dates use the freshly computed rows.
        combined = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if not existing_df.empty else new_df.copy()
        )
        combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        combined.sort_values("Date", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        if not existing_df.empty:
            for missing_col in set(existing_df.columns) - set(combined.columns):
                combined[missing_col] = 0.0
                _log(f"[Schema] Back-filled column from existing data: {missing_col}")

        combined = compute_rolling_features(combined)

        # ── Schema validation ──────────────────────────────────────────────────
        if not existing_df.empty:
            existing_pub = {c for c in existing_df.columns if not c.startswith(_GC_PFX)}
            combined_pub = {c for c in combined.columns    if not c.startswith(_GC_PFX)}
            removed = existing_pub - combined_pub
            if removed:
                msg = (f"!! SCHEMA ERROR: {len(removed)} column(s) removed from parquet "
                       f"→ {sorted(removed)}")
                _log(msg)
                return {"statusCode": 500, "body": msg}
            added = combined_pub - existing_pub
            if added:
                _log(f"[Schema] {len(added)} new column(s) added → {sorted(added)}")

        # ── Column audit ───────────────────────────────────────────────────────
        if not existing_df.empty:
            _audit_columns(existing_df, new_rows, combined)

        # ── Save ───────────────────────────────────────────────────────────────
        save_parquet_to_mongo(combined)
        gc.collect()

        # ── Final summary ──────────────────────────────────────────────────────
        summary = {
            "dates_processed":    [str(r["Date"]) for r in new_rows],
            "dates_skipped":      skipped_dates,
            "new_rows":           len(new_rows),
            "new_rows_cols":      len(new_df.columns),
            "new_rows_pub_cols":  len([c for c in new_df.columns if not c.startswith(_GC_PFX)]),
            "total_rows":         len(combined),
            "total_cols":         len(combined.columns),
            "total_pub_cols":     len([c for c in combined.columns if not c.startswith(_GC_PFX)]),
            "last_date":          str(combined["Date"].max()),
        }

        _log_section("PIPELINE COMPLETE")
        _log(json.dumps(summary, indent=2))

        # CHANGE 2: Trigger inference Lambda asynchronously.
        # Fires-and-forgets — pipeline success is not contingent on this.
        try:
            lambda_client = boto3.client(
                "lambda", region_name=os.environ.get("AWS_REGION", "ap-south-1")
            )
            invoke_resp = lambda_client.invoke(
                FunctionName="Range-Model-Predictor",
                InvocationType="Event",  # async — does not wait for response
                Payload=json.dumps({
                    "source":    "nifty-feature-pipeline",
                    "last_date": str(combined["Date"].max()),
                    "dates_processed": [str(r["Date"]) for r in new_rows],
                }),
            )
            _log(f"[Trigger] Range-Model-Predictor invoked async, "
                 f"StatusCode={invoke_resp['StatusCode']}")
        except Exception as trigger_exc:
            # Log but do NOT fail the pipeline — data is already saved
            _log_error("trigger Range-Model-Predictor", trigger_exc)

        return {"statusCode": 200, "body": json.dumps(summary)}

    except Exception as exc:
        _log_error("handler (top-level)", exc)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error":     str(sys.exc_info()[1]),
                "traceback": traceback.format_exc(),
            }),
        }