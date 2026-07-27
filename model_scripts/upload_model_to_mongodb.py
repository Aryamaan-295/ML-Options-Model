#!/usr/bin/env python3
# ================================================================================
# upload_to_mongodb.py
# Run once to push model artefacts and/or processed data to MongoDB Atlas.
#
# Usage
# -----
#   Set the flag variables in the "RUN CONFIGURATION" section below, then:
#       python upload_to_mongodb.py
#
# Expected file structure (relative to this script):
#   ./model_scripts/upload_to_mongodb.py          <- this file
#   ./model_scripts/final_model/*.pt              <- model weight files
#   ./model_scripts/final_model/*.onnx            <- ONNX files (if export succeeded)
#   ./model_scripts/final_model/*_meta.json       <- metadata JSON
#   ./processed_data/nifty_master_file.parquet    <- market data
#
# MongoDB layout  (db = "deployments"):
#   model_weights              — metadata JSON documents (one per training run)
#   model_onnx.files/.chunks   — GridFS: ONNX binaries  (used by Lambda)
#   model_pt.files/.chunks     — GridFS: PT checkpoints (backup)
#   processed_data.files/.chunks — GridFS: master parquet
#   inference_results          — written by Lambda at inference time
#
# Requirements: pip install "pymongo[srv]" dnspython
# ================================================================================

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# ================================================================================
# RUN CONFIGURATION — edit these flags before running
# ================================================================================

# Your MongoDB Atlas connection string.
# Can also be set via the MONGO_URI environment variable (takes priority).
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")

# ── What to upload ────────────────────────────────────────────────────────────
# Set True/False independently 

UPLOAD_MODEL_WEIGHTS = True
# Upload PT + ONNX files and the metadata JSON for the training run below.

UPLOAD_PROCESSED_DATA = True
# Upload the master parquet file to GridFS.

# ── Which training run to upload ──────────────────────────────────────────────
# Timestamp printed at the end of Cell 9, e.g. "20260322_170748".
# Only used when UPLOAD_MODEL_WEIGHTS = True.
MODEL_TIMESTAMP = "20260322_172626"

# ── File paths ────────────────────────────────────────────────────────────────
# _HERE is the model_scripts/ directory (where this script lives).
# Paths below match the documented file structure — only change if your layout differs.
_HERE           = Path(__file__).resolve().parent          # .../model_scripts/
FINAL_MODEL_DIR = _HERE / "final_model"                    # .../model_scripts/final_model/
DATA_FILE       = _HERE.parent / "processed_data" / "nifty_master_file.parquet"

# ================================================================================
# END OF CONFIGURATION
# ================================================================================


try:
    import gridfs
    import pymongo
except ImportError:
    sys.exit(
        "pymongo not installed.\n"
        "Run:  pip install \'pymongo[srv]\' dnspython"
    )

DB_NAME = "deployments"


def _connect() -> pymongo.MongoClient:
    uri = os.environ.get("MONGO_URI", MONGO_URI)
    if "YOUR_MONGODB" in uri:
        sys.exit(
            "ERROR: Set the MONGO_URI constant at the top of this script "
            "(or export the MONGO_URI environment variable) before running."
        )
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=15_000)
    client[DB_NAME].command("ping")
    print("Connected to MongoDB Atlas.")
    return client


def _ensure_indexes(db: pymongo.database.Database) -> None:
    db["model_weights"].create_index(
        [("timestamp", pymongo.DESCENDING)], name="ts_desc"
    )
    db["model_weights"].create_index(
        [("schema_version", pymongo.ASCENDING)], name="schema_asc"
    )
    db["inference_results"].create_index(
        [("model_timestamp", pymongo.ASCENDING),
         ("Prediction_Date",  pymongo.ASCENDING)],
        unique=True, name="model_date_unique",
    )
    print("Indexes ensured.")


def _upload_gridfs(
    db: pymongo.database.Database,
    bucket: str,
    file_path: Path,
    filename: str,
    metadata: dict,
) -> str:
    """
    Upload file_path to GridFS bucket under filename.
    Safe upload order:
      1. Upload new file under a temporary name.
      2. Verify the upload is readable (read back first byte).
      3. Only then delete the previous file(s) with the real filename.
      4. Re-upload under the real filename and delete the temp copy.
    Old files are never deleted if any step before (3) fails, so a failed
    upload never results in data loss.
    """
    fs       = gridfs.GridFS(db, collection=bucket)
    tmp_name = f"__tmp__{filename}"
    new_oid  = None
    tmp_oid  = None

    try:
        # Step 1: upload to temp name
        with open(file_path, "rb") as fh:
            tmp_oid = fs.put(fh, filename=tmp_name, metadata=metadata)

        # Step 2: verify readability — read first 4 bytes back
        tmp_gf = fs.get(tmp_oid)
        _ = tmp_gf.read(4)
        tmp_gf.close()

        # Step 3: safe to delete old file(s) now
        for old_gf in fs.find({"filename": filename}):
            fs.delete(old_gf._id)

        # Step 4: upload under real name, delete temp
        with open(file_path, "rb") as fh:
            new_oid = fs.put(fh, filename=filename, metadata=metadata)
        fs.delete(tmp_oid)
        tmp_oid = None

        return str(new_oid)

    except Exception:
        # Clean up any partial temp upload; original files are untouched
        if tmp_oid is not None:
            try:
                fs.delete(tmp_oid)
            except Exception:
                pass
        raise


def upload_model_weights(db: pymongo.database.Database, timestamp: str) -> None:
    meta_path = FINAL_MODEL_DIR / f"nifty_1d_{timestamp}_meta.json"
    if not meta_path.exists():
        sys.exit(
            f"ERROR: Metadata file not found:\n  {meta_path}\n"
            "Check MODEL_TIMESTAMP and FINAL_MODEL_DIR."
        )

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    print(f"\n── Model weights upload: {timestamp} ──")

    onnx_uploaded = 0
    pt_uploaded   = 0

    for member in meta.get("model_files", []):
        midx      = member["model_index"]
        is_anchor = member.get("is_anchor", False)
        tag       = " (anchor)" if is_anchor else ""
        file_meta = {
            "timestamp"  : timestamp,
            "model_index": midx,
            "is_anchor"  : is_anchor,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # ONNX
        onnx_name   = member.get("onnx_file")
        onnx_status = member.get("onnx_export_status", "unknown")

        if onnx_name and onnx_status != "failed":
            onnx_path = FINAL_MODEL_DIR / onnx_name
            if onnx_path.exists():
                oid     = _upload_gridfs(db, "model_onnx", onnx_path, onnx_name,
                                         {**file_meta, "format": "onnx"})
                size_kb = onnx_path.stat().st_size / 1024
                print(f"  [onnx] model {midx}{tag}  {onnx_name}  ({size_kb:.1f} KB)  _id={oid}")
                onnx_uploaded += 1
            else:
                print(f"  [warn] ONNX listed in metadata but not found: {onnx_path}")
        else:
            print(f"  [skip] model {midx} ONNX -- file={onnx_name!r}  status={onnx_status}")

        # PT
        pt_name = member.get("pt_file")
        if pt_name:
            pt_path = FINAL_MODEL_DIR / pt_name
            if pt_path.exists():
                oid     = _upload_gridfs(db, "model_pt", pt_path, pt_name,
                                         {**file_meta, "format": "pt"})
                size_kb = pt_path.stat().st_size / 1024
                print(f"  [pt  ] model {midx}{tag}  {pt_name}  ({size_kb:.1f} KB)  _id={oid}")
                pt_uploaded += 1
            else:
                print(f"  [warn] PT file listed in metadata but not found: {pt_path}")

    # Metadata document
    coll    = db["model_weights"]
    removed = coll.delete_many({"timestamp": timestamp}).deleted_count
    if removed:
        print(f"  [meta] Replaced {removed} existing metadata doc(s).")
    coll.insert_one(meta)
    print(f"  [meta] Inserted metadata document.")
    print(f"\n  Model upload complete | ONNX={onnx_uploaded} | PT={pt_uploaded} | ts={timestamp}")

    if pt_uploaded > 0 and onnx_uploaded == 0:
        print(
            "\n  WARNING: No ONNX files uploaded (export failed in Cell 9).\n"
            "  Lambda inference requires ONNX. Apply the dynamo=False patch to\n"
            "  Cell 9, retrain, then re-run this upload script."
        )


def upload_processed_data(db: pymongo.database.Database) -> None:
    if not DATA_FILE.exists():
        sys.exit(
            f"ERROR: Parquet file not found:\n  {DATA_FILE}\n"
            "Check the DATA_FILE path at the top of this script."
        )

    size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
    print(f"\n── Processed data upload: {DATA_FILE.name}  ({size_mb:.2f} MB) ──")

    oid = _upload_gridfs(
        db,
        "processed_data",
        DATA_FILE,
        DATA_FILE.name,
        {
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "source"     : str(DATA_FILE),
        },
    )
    print(f"  Uploaded {DATA_FILE.name}  _id={oid}")


def print_summary(db: pymongo.database.Database) -> None:
    coll = db["model_weights"]
    docs = list(coll.find(
        {},
        {"timestamp": 1, "dataset_rows": 1, "last_dataset_date": 1,
         "model_files": 1, "_id": 0},
        sort=[("timestamp", pymongo.DESCENDING)],
        limit=10,
    ))
    if not docs:
        print("\nNo model runs found in deployments.model_weights.")
        return

    print(f"\n{chr(8212)*72}")
    print(f"  Stored model runs (most recent first):")
    print(f"  {'Timestamp':<20} {'Rows':>6}  {'Last Date':<12}  {'ONNX':>5}  {'PT':>4}")
    print(f"{chr(8212)*72}")
    for d in docs:
        n_onnx = sum(
            1 for m in d.get("model_files", [])
            if m.get("onnx_file") and m.get("onnx_export_status") != "failed"
        )
        n_pt = sum(1 for m in d.get("model_files", []) if m.get("pt_file"))
        print(
            f"  {d.get('timestamp', 'N/A'):<20} "
            f"{d.get('dataset_rows', '?'):>6}  "
            f"{str(d.get('last_dataset_date', '?')):<12}  "
            f"{n_onnx:>5}  {n_pt:>4}"
        )
    print(f"{chr(8212)*72}")

    fs    = gridfs.GridFS(db, collection="processed_data")
    files = list(fs.find({}, sort=[("uploadDate", -1)], limit=1))
    if files:
        f = files[0]
        print(f"\n  Latest processed data: {f.filename}  "
              f"({f.length / (1024*1024):.2f} MB)  "
              f"uploaded {f.upload_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        print("\n  No processed data in deployments.processed_data.")


def main() -> None:
    if not UPLOAD_MODEL_WEIGHTS and not UPLOAD_PROCESSED_DATA:
        print(
            "Nothing to do -- both UPLOAD_MODEL_WEIGHTS and UPLOAD_PROCESSED_DATA\n"
            "are False. Edit the flags at the top of this script."
        )
        return

    client = _connect()
    db     = client[DB_NAME]
    _ensure_indexes(db)

    if UPLOAD_MODEL_WEIGHTS:
        upload_model_weights(db, MODEL_TIMESTAMP)

    if UPLOAD_PROCESSED_DATA:
        upload_processed_data(db)

    print_summary(db)
    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()