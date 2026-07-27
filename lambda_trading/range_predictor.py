# ================================================================================
# lambda_function.py
# NIFTY Inference Lambda
#
# Runtime        : Python 3.11 x86_64
# Memory         : 512 MB
# Ephemeral /tmp : 512 MB (default)
# Timeout        : 300 s
#
# Env vars:
#   SSM_MONGO_PARAM    path to MongoDB URI in SSM  (default: /fyers/MONGO_URI)
#   STORE_RESULTS      "true" to upsert into inference_results (default: true)
#
# SSM parameters:
#   /fyers/MONGO_URI            MongoDB Atlas connection string  (SecureString)
#   /fyers/TELEGRAM_BOT_TOKEN   Telegram bot token              (SecureString)
#   /fyers/TELEGRAM_CHAT_ID     Telegram chat/channel ID        (SecureString)
#
# Lambda layers:
#   Layer 1 : ONNX-Polars-Dependencies
#             (onnxruntime==1.16.3, polars==0.20.31, numpy==1.26.4, requests==2.31.0)
#   pymongo + dnspython bundled inside the function .zip
#   boto3 built into the Lambda runtime
#
# Date convention:
#   Prediction_Date    = date T  (last data row used as model input)
#   Predicted_for_Date = date T+1 (next trading weekday being forecast)
# ================================================================================

from __future__ import annotations

import gc
import json
import math
import os
import traceback
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Any

import boto3
import numpy as np
import requests

# ── Module-level singletons ───────────────────────────────────────────────────
_mongo_client = None
_ssm_client   = None
_mongo_uri    = None
_ort          = None
_pymongo      = None
_gridfs       = None
_polars       = None
_lambda_client= None

def _lambda_client_singleton():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client(
            "lambda",
            region_name=os.environ.get("AWS_REGION", "ap-south-1"),
        )
    return _lambda_client


# ── Lazy importers ────────────────────────────────────────────────────────────

def _get_ort():
    global _ort
    if _ort is None:
        import onnxruntime as ort
        _ort = ort
    return _ort


def _get_mongo_libs():
    global _pymongo, _gridfs
    if _pymongo is None:
        import pymongo
        import gridfs
        _pymongo = pymongo
        _gridfs  = gridfs
    return _pymongo, _gridfs


def _get_polars():
    global _polars
    if _polars is None:
        import polars as pl
        _polars = pl
    return _polars


# ── AWS / SSM ─────────────────────────────────────────────────────────────────

def _ssm_client_singleton():
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client(
            "ssm",
            region_name=os.environ.get("AWS_REGION", "ap-south-1"),
        )
    return _ssm_client


def get_secret(param_name: str, decrypt: bool = True) -> str:
    full_name = f"/fyers/{param_name}" if not param_name.startswith("/") else param_name
    resp = _ssm_client_singleton().get_parameter(
        Name=full_name, WithDecryption=decrypt
    )
    return resp["Parameter"]["Value"]


def _load_ssm_params() -> None:
    global _mongo_uri
    if _mongo_uri is None:
        param_name = os.environ.get("SSM_MONGO_PARAM", "/fyers/MONGO_URI")
        _mongo_uri = get_secret(param_name)
        print(f"[SSM] Loaded {param_name}")

def invoke_basket_analyser() -> None:
    """
    Triggers the Fyers-Basket-Analyser Lambda asynchronously (Event).
    """
    print("[Trigger] Attempting to invoke Fyers-Basket-Analyser...")
    try:
        client = _lambda_client_singleton()
        
        client.invoke(
            FunctionName="Fyers-Basket-Analyser",
            InvocationType="Event",
        )
        print("[Trigger] Fyers-Basket-Analyser invoked successfully.")
    except Exception as e:
        print(f"[Trigger] FAILED to invoke Basket Analyser: {type(e).__name__}: {e}")


# ── MongoDB client ────────────────────────────────────────────────────────────

def _mongo_client_singleton():
    global _mongo_client
    if _mongo_client is None:
        pymongo, _ = _get_mongo_libs()
        _load_ssm_params()
        _mongo_client = pymongo.MongoClient(
            _mongo_uri,
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=60_000,
        )
        print("[MongoDB] Client created")
    return _mongo_client


# ── Date helpers ──────────────────────────────────────────────────────────────

def _next_trading_day(d: date_type) -> date_type:
    next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)
    return next_d


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram_alert(message: str) -> None:
    print("[Telegram] Attempting to send message...")
    try:
        bot_token = get_secret("TELEGRAM_BOT_TOKEN", decrypt=True)
        print(f"[Telegram] Token fetched, length={len(bot_token)}")
    except Exception as e:
        print(f"[Telegram] FAILED to fetch TELEGRAM_BOT_TOKEN: {type(e).__name__}: {e}")
        return

    try:
        chat_id = get_secret("TELEGRAM_CHAT_ID", decrypt=True)
        print(f"[Telegram] Chat ID fetched: {chat_id}")
    except Exception as e:
        print(f"[Telegram] FAILED to fetch TELEGRAM_CHAT_ID: {type(e).__name__}: {e}")
        return

    if not bot_token or not chat_id:
        print("[Telegram] FAILED — empty token or chat_id after fetch")
        return

    url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    print(f"[Telegram] POSTing to {url[:60]}... chat_id={chat_id}")
    try:
        resp = requests.post(url, json=payload, timeout=10)
        print(f"[Telegram] Response HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[Telegram] POST failed — {type(e).__name__}: {e}")


def _format_telegram_message(latest_row: dict, meta: dict, n_new: int) -> str:
    bands     = meta["output_spec"]["ci_bands"]
    outer_pct = bands["outer"]["ci_pct"]
    inner_pct = bands["inner"]["ci_pct"]

    pred_for    = latest_row.get("Predicted_for_Date", "N/A")
    base_price  = latest_row.get("Base_Price", "N/A")
    mean_raw    = latest_row.get("Mean_Predicted_Raw", "N/A")
    mean_price  = latest_row.get("Mean_Predicted_Price", "N/A")
    lo_outer    = latest_row.get(f"Lower_{outer_pct}pct_CI_Price", "N/A")
    hi_outer    = latest_row.get(f"Upper_{outer_pct}pct_CI_Price", "N/A")
    lo_inner    = latest_row.get(f"Lower_{inner_pct}pct_CI_Price", "N/A")
    hi_inner    = latest_row.get(f"Upper_{inner_pct}pct_CI_Price", "N/A")
    outer_width = latest_row.get(f"Band_{outer_pct}pct_Width_Pts", "N/A")
    inner_width = latest_row.get(f"Band_{inner_pct}pct_Width_Pts", "N/A")

    direction = "BULLISH" if isinstance(mean_raw, float) and mean_raw > 0 else "BEARISH"

    def fmt(v):
        return f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)

    def fmtw(v):
        return f"{v:,.0f} pts" if isinstance(v, (int, float)) else str(v)

    label = "NEW FORECAST" if n_new > 0 else "LATEST STORED FORECAST"

    # Logic for direction color/emoji
    if direction == "BULLISH":
        dir_display = "🟢 <b>BULLISH</b>"
    else:
        dir_display = "🔴 <b>BEARISH</b>"

    return (
        f"<b>NIFTY {label}</b>\n"
        f"Date: <code>{pred_for}</code>\n"
        f"------------------------------------------------------\n"
        f"Direction      : {dir_display}\n"
        f"Base (T close) : <code>{fmt(base_price)}</code>\n"
        f"Mean Forecast  : <code>{fmt(mean_price)}</code>\n"
        f"\n"
        f"<u>{outer_pct}% CI :</u> \n <code>[{fmt(lo_outer)}, {fmt(hi_outer)}]</code> (<i>{fmtw(outer_width )}</i>)\n"
        f"<u>{inner_pct}% CI :</u> \n <code>[{fmt(lo_inner)}, {fmt(hi_inner)}]</code> (<i>{fmtw(inner_width )}</i>)\n"
        f"\n"
        f"Model : <code>{meta['timestamp']}</code>\n"
        f"Run   : <i>{n_new} new row(s) predicted</i>"
    )

# ── GridFS helpers ────────────────────────────────────────────────────────────

def _gridfs_download(db, bucket_name: str, filename: str, dest: str) -> int:
    _, gridfs_mod = _get_mongo_libs()
    bucket = gridfs_mod.GridFS(db, collection=bucket_name)
    gf     = bucket.find_one({"filename": filename})
    if gf is None:
        raise FileNotFoundError(
            f"GridFS file '{filename}' not found in bucket '{bucket_name}'"
        )
    data = gf.read()
    with open(dest, "wb") as fh:
        fh.write(data)
    return len(data)


def _cached_onnx_path(db, onnx_filename: str, run_timestamp: str) -> str:
    path = f"/tmp/{onnx_filename}"
    if os.path.exists(path):
        print(f"[Cache] ONNX hit: {onnx_filename}")
        return path
    print(f"[Cache] Downloading ONNX: {onnx_filename}")
    nbytes = _gridfs_download(db, "model_onnx", onnx_filename, path)
    print(f"[Cache] Saved {nbytes / 1024:.1f} KB -> {path}")
    return path


def _cached_parquet_path(db) -> str:
    _, gridfs_mod = _get_mongo_libs()
    bucket = gridfs_mod.GridFS(db, collection="processed_data")
    files  = list(bucket.find({}, sort=[("uploadDate", -1)], limit=1))
    if not files:
        raise RuntimeError(
            "No file found in deployments.processed_data GridFS."
        )
    latest    = files[0]
    file_id   = str(latest._id)
    parquet_p = "/tmp/nifty_master_file.parquet"
    marker    = f"/tmp/_data_id_{file_id}"

    if os.path.exists(marker) and os.path.exists(parquet_p):
        print("[Cache] Parquet hit")
        return parquet_p

    print(f"[Cache] Downloading parquet ({latest.length / 1024:.1f} KB)")
    data = latest.read()
    with open(parquet_p, "wb") as fh:
        fh.write(data)
    for old in (f for f in os.listdir("/tmp") if f.startswith("_data_id_")):
        os.remove(f"/tmp/{old}")
    open(marker, "w").close()
    print("[Cache] Parquet saved")
    return parquet_p


# ── Metadata ──────────────────────────────────────────────────────────────────

def _load_metadata(db) -> dict:
    coll = db["model_weights"]
    doc  = coll.find_one(
        {"schema_version": {"$exists": True}},
        sort=[("timestamp", -1)],
    )
    if doc is None:
        raise RuntimeError("No metadata document found in deployments.model_weights.")
    doc.pop("_id", None)
    print(
        f"[Metadata] timestamp={doc['timestamp']} | "
        f"rows={doc['dataset_rows']} | "
        f"last_date={doc.get('last_dataset_date')}"
    )
    return doc


# ── Existing result dates ─────────────────────────────────────────────────────

def _get_existing_result_dates(db, model_timestamp: str) -> set[str]:
    coll = db["inference_results"]
    docs = coll.find(
        {"model_timestamp": model_timestamp},
        {"Prediction_Date": 1, "_id": 0},
    )
    existing = {d["Prediction_Date"] for d in docs if "Prediction_Date" in d}
    if existing:
        print(f"[Results] {len(existing)} existing rows, latest={sorted(existing)[-1]}")
    else:
        print("[Results] No existing rows — full run")
    return existing


# ── Fetch latest stored row ───────────────────────────────────────────────────

def _get_latest_stored_row(db, model_timestamp: str) -> dict | None:
    print(f"[Results] Fetching latest stored row for model_timestamp={model_timestamp}")
    coll = db["inference_results"]
    doc  = coll.find_one(
        {"model_timestamp": model_timestamp},
        sort=[("Prediction_Date", -1)],
    )
    if doc is None:
        print("[Results] No stored rows found")
        return None
    doc.pop("_id", None)
    print(
        f"[Results] Found: Prediction_Date={doc.get('Prediction_Date')} "
        f"Predicted_for_Date={doc.get('Predicted_for_Date')}"
    )
    return doc


# ── EWMA (numpy) ──────────────────────────────────────────────────────────────

def _ewma_numpy(arr: np.ndarray, span: int) -> np.ndarray:
    alpha  = 2.0 / (span + 1.0)
    decay  = 1.0 - alpha
    result = np.empty(len(arr), dtype=np.float32)
    num    = float(arr[0])
    den    = 1.0
    result[0] = num
    for i in range(1, len(arr)):
        num       = float(arr[i]) + decay * num
        den       = 1.0           + decay * den
        result[i] = num / den
    return result


# ── Data loading & preprocessing ─────────────────────────────────────────────

def _load_and_scale(
    db, meta: dict
) -> tuple[np.ndarray, list[date_type], np.ndarray, np.ndarray]:
    pl           = _get_polars()
    parquet_p    = _cached_parquet_path(db)
    feat_reg     = meta["feature_registry"]
    feature_cols = feat_reg["feature_cols"]
    prep         = meta["preprocessing"]
    ewma_span    = meta["vol_scaling"]["ewma_span"]

    df = pl.read_parquet(parquet_p)

    required = set(feature_cols + ["Date", "Spot_Close", "Realized_Vol_Ann"])
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {sorted(missing)}")

    X_raw      = df.select(feature_cols).to_numpy().astype(np.float32)
    spot_close = df["Spot_Close"].to_numpy().astype(np.float64)
    realized_v = df["Realized_Vol_Ann"].to_numpy().astype(np.float32)

    raw_dates = df["Date"].to_list()
    dates: list[date_type] = [
        d.date() if hasattr(d, "date") else
        datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        for d in raw_dates
    ]

    vol_ewma = _ewma_numpy(realized_v, ewma_span)

    var_idx     = feat_reg["var_feature_indices"]
    all_dir_idx = feat_reg["all_dir_indices"]
    var_mean    = np.array(prep["var_features"]["mean"],  dtype=np.float32)
    var_scale   = np.array(prep["var_features"]["scale"], dtype=np.float32)
    dir_scale   = np.array(prep["dir_features"]["scale"], dtype=np.float32)

    X_scaled = np.zeros_like(X_raw)
    X_scaled[:, var_idx]     = (X_raw[:, var_idx] - var_mean) / var_scale
    X_scaled[:, all_dir_idx] = X_raw[:, all_dir_idx] / dir_scale

    print(
        f"[Data] {len(dates)} rows | "
        f"features={X_scaled.shape[1]} | "
        f"range={dates[0]} -> {dates[-1]}"
    )
    return X_scaled, dates, spot_close, vol_ewma


# ── Output column name builder ────────────────────────────────────────────────

def _col_names(meta: dict) -> dict[str, tuple[str, str]]:
    bands     = meta["output_spec"]["ci_bands"]
    outer_pct = bands["outer"]["ci_pct"]
    inner_pct = bands["inner"]["ci_pct"]
    return {
        bands["outer"]["lower_name"]: (f"Lower_{outer_pct}pct_CI_Price", f"Lower_{outer_pct}pct_CI_Raw"),
        bands["inner"]["lower_name"]: (f"Lower_{inner_pct}pct_CI_Price", f"Lower_{inner_pct}pct_CI_Raw"),
        "expected_return"            : ("Mean_Predicted_Price",           "Mean_Predicted_Raw"),
        bands["inner"]["upper_name"]: (f"Upper_{inner_pct}pct_CI_Price", f"Upper_{inner_pct}pct_CI_Raw"),
        bands["outer"]["upper_name"]: (f"Upper_{outer_pct}pct_CI_Price", f"Upper_{outer_pct}pct_CI_Raw"),
    }


# ── Core inference ────────────────────────────────────────────────────────────

def _run_inference(
    meta: dict,
    db,
    existing_dates: set[str],
) -> list[dict]:
    ort          = _get_ort()
    seq_len      = meta["config"]["seq_len_1d"]
    output_names = meta["output_spec"]["output_names"]
    col_map      = _col_names(meta)
    last_date_s  = meta.get("last_dataset_date")
    bands        = meta["output_spec"]["ci_bands"]
    outer_pct    = bands["outer"]["ci_pct"]
    inner_pct    = bands["inner"]["ci_pct"]

    X_scaled, dates, spot_close, vol_ewma = _load_and_scale(db, meta)

    if last_date_s:
        last_train = datetime.strptime(last_date_s, "%Y-%m-%d").date()
    else:
        last_train = dates[0]

    # CHANGE: Identify all valid post-training indices (seq_len satisfied),
    # then always force-include the last 2 regardless of existing_dates.
    # Older indices still skip if already in existing_dates.
    all_valid_idx = [
        i for i, d in enumerate(dates)
        if d > last_train and i >= seq_len - 1
    ]

    if not all_valid_idx:
        print("[Inference] No valid post-training dates in data")
        return []

    # Last 2 valid indices are always rerun; rest skip if already stored
    force_rerun = set(all_valid_idx[-2:])
    valid_idx   = [
        i for i in all_valid_idx
        if str(dates[i]) not in existing_dates or i in force_rerun
    ]

    if not valid_idx:
        print("[Inference] No new dates to predict")
        return []

    rerun_dates = [str(dates[i]) for i in valid_idx if i in force_rerun]
    print(
        f"[Inference] {len(valid_idx)} rows to predict: "
        f"{dates[valid_idx[0]]} -> {dates[valid_idx[-1]]} | "
        f"force-rerun (last 2): {rerun_dates}"
    )

    ensemble: dict[int, list[list[float]]] = {
        i: [[] for _ in range(len(output_names))]
        for i in valid_idx
    }

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads     = 2
    sess_opts.inter_op_num_threads     = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

    usable = 0
    for member in meta["model_files"]:
        onnx_file = member.get("onnx_file")
        status    = member.get("onnx_export_status", "unknown")

        if not onnx_file or status == "failed":
            print(f"[Model {member['model_index']}] Skipped (status={status})")
            continue

        onnx_path = _cached_onnx_path(db, onnx_file, meta["timestamp"])
        sess      = ort.InferenceSession(
            onnx_path, sess_opts, providers=["CPUExecutionProvider"]
        )

        for i in valid_idx:
            window = X_scaled[i - seq_len + 1 : i + 1]
            x_np   = window[np.newaxis].astype(np.float32)
            outs   = sess.run(None, {"feature_sequence": x_np})
            for k, arr in enumerate(outs):
                ensemble[i][k].append(float(arr[0, 0]))

        del sess
        gc.collect()
        usable += 1
        print(f"[Model {member['model_index']}] Inference complete")

    if usable == 0:
        raise RuntimeError("No usable ONNX models found.")

    results: list[dict] = []

    for i in valid_idx:
        vol           = float(vol_ewma[i])
        base_p        = float(spot_close[i])
        pred_date     = dates[i]
        predicted_for = _next_trading_day(pred_date)

        Qm_raw = [
            float(np.mean(ensemble[i][k])) * vol
            if ensemble[i][k] else float("nan")
            for k in range(len(output_names))
        ]

        row: dict[str, Any] = {
            "Prediction_Date"   : str(pred_date),
            "Predicted_for_Date": str(predicted_for),
            "Base_Price"        : round(base_p, 2),
            "Implied_Vol_Scaler": round(vol, 6),
        }

        for k, out_name in enumerate(output_names[:5]):
            if out_name in col_map:
                price_col, raw_col = col_map[out_name]
                row[raw_col]   = round(Qm_raw[k], 6)
                row[price_col] = round(base_p * math.exp(Qm_raw[k]), 2)

        if len(output_names) > 5:
            raw_aux = ensemble[i][5]
            row["Aux_Pred_Raw"] = (
                round(float(np.mean(raw_aux)), 6) if raw_aux else None
            )

        lo_outer_p = row.get(f"Lower_{outer_pct}pct_CI_Price")
        hi_outer_p = row.get(f"Upper_{outer_pct}pct_CI_Price")
        lo_inner_p = row.get(f"Lower_{inner_pct}pct_CI_Price")
        hi_inner_p = row.get(f"Upper_{inner_pct}pct_CI_Price")

        if lo_outer_p is not None and hi_outer_p is not None:
            row[f"Band_{outer_pct}pct_Width_Pts"] = round(hi_outer_p - lo_outer_p, 2)
        if lo_inner_p is not None and hi_inner_p is not None:
            row[f"Band_{inner_pct}pct_Width_Pts"] = round(hi_inner_p - lo_inner_p, 2)

        results.append(row)

    print(f"[Inference] Built {len(results)} prediction rows")
    return results


# ── Store results ─────────────────────────────────────────────────────────────

def _store_results(db, results: list[dict], meta: dict) -> str:
    coll = db["inference_results"]
    ts   = meta["timestamp"]
    for row in results:
        coll.update_one(
            {"model_timestamp": ts, "Prediction_Date": row["Prediction_Date"]},
            {"$set": {
                **row,
                "model_timestamp": ts,
                "stored_at"      : datetime.utcnow().isoformat(),
            }},
            upsert=True,
        )
    msg = f"Upserted {len(results)} rows into deployments.inference_results"
    print(f"[MongoDB] {msg}")
    return msg


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    print("=" * 50)
    print("NIFTY Inference Lambda — invocation start")
    print("=" * 50)

    tg_message  = None
    meta        = None
    n_new       = 0
    error_text  = None
    response    = None

    try:
        _load_ssm_params()

        client = _mongo_client_singleton()
        db     = client["deployments"]

        meta           = _load_metadata(db)
        model_ts       = meta["timestamp"]
        existing_dates = _get_existing_result_dates(db, model_ts)

        results   = _run_inference(meta, db, existing_dates)
        store_msg = ""
        if results:
            store_msg = _store_results(db, results, meta)

        n_new = len(results)

        # Telegram reports only the last row (latest prediction), same as before
        if results:
            report_row = results[-1]
            print(f"[Handler] {n_new} prediction(s) stored")
        else:
            print("[Handler] No new predictions — fetching latest stored row")
            report_row = _get_latest_stored_row(db, model_ts)

        if report_row is not None and meta is not None:
            tg_message = _format_telegram_message(report_row, meta, n_new)
        else:
            tg_message = (
                f"NIFTY Inference ran but no rows found to report.\n"
                f"Model: {meta['timestamp'] if meta else 'unknown'}"
            )

        invoke_basket_analyser()

        ci_bands = meta["output_spec"]["ci_bands"]
        if results:
            response = {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message"          : f"Predicted and stored {n_new} row(s)",
                        "model_timestamp"  : model_ts,
                        "last_dataset_date": meta.get("last_dataset_date"),
                        "n_new_predictions": n_new,
                        "date_range": {
                            "prediction_date_from"   : results[0]["Prediction_Date"],
                            "prediction_date_to"     : results[-1]["Prediction_Date"],
                            "predicted_for_date_from": results[0]["Predicted_for_Date"],
                            "predicted_for_date_to"  : results[-1]["Predicted_for_Date"],
                        },
                        "column_guide": {
                            "outer_ci_pct"   : ci_bands["outer"]["ci_pct"],
                            "inner_ci_pct"   : ci_bands["inner"]["ci_pct"],
                            "note"           : (
                                "_Raw = log returns. "
                                "_Price = Nifty index points. "
                                "Band_Xpct_Width_Pts = CI width in pts."
                            ),
                        },
                        "store_msg"  : store_msg,
                        "predictions": results,
                    },
                    default=str,
                ),
            }
        else:
            response = {
                "statusCode": 200,
                "body": json.dumps({
                    "message"          : (
                        "No new dates to predict \u2014 all post-training dates "
                        "already stored in inference_results."
                    ),
                    "last_dataset_date": meta.get("last_dataset_date"),
                    "model_timestamp"  : model_ts,
                    "existing_count"   : len(existing_dates),
                }),
            }

    except Exception:
        full_tb    = traceback.format_exc()
        error_text = full_tb
        print("=" * 50)
        print("INFERENCE FAILED")
        print(full_tb)
        print("=" * 50)
        tg_message = (
            f"NIFTY Inference FAILED - "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n\n"
            f"{full_tb[-1000:]}"
        )
        response = {
            "statusCode": 500,
            "body": json.dumps({
                "error"    : "Inference failed",
                "traceback": full_tb,
            }),
        }

    # ── Always send Telegram, unconditionally ─────────────────────────────────
    print("[Handler] Sending Telegram alert (unconditional)")
    try:
        if tg_message:
            send_telegram_alert(tg_message)
        else:
            send_telegram_alert("NIFTY Inference ran but no message was composed.")
    except Exception as tg_err:
        print(f"[Telegram] Outer send failed: {type(tg_err).__name__}: {tg_err}")

    print("=" * 50)
    print("[Handler] Invocation complete")
    print("=" * 50)
    return response