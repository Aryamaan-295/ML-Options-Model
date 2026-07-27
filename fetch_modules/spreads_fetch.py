import sys
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
from datetime import datetime, timedelta
import calendar
import logging
from pathlib import Path
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_paths():
    """Resolves absolute paths safely using pathlib."""
    base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / '.env'
    data_dir = base_dir / 'data' / 'NIFTY_spreads'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(env_path), str(data_dir)

def get_mongo_client(env_path):
    """Loads credentials and connects to MongoDB bypassing local SSL chain issues."""
    load_dotenv(dotenv_path=env_path)
    conn_string = os.getenv("MONGODB_CONNECTION_STRING")
    
    if not conn_string:
        logging.error("MONGODB_CONNECTION_STRING not found in .env file.")
        sys.exit(1)
        
    try:
        client = MongoClient(
            conn_string, 
            tls=True, 
            tlsAllowInvalidCertificates=True, 
            serverSelectionTimeoutMS=10000
        )
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB Atlas!")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

def parse_expiry_to_str(expiry_val):
    """Parses mixed NSE expiry formats to a standardized IST datetime string."""
    if pd.isna(expiry_val):
        return expiry_val
        
    expiry_str = str(expiry_val).strip().upper()
    
    try:
        # If it's already a standard datetime string, leave it
        if "-" in expiry_str and ":" in expiry_str:
            return expiry_str
            
        # Format: YYMMM (e.g., "26MAR") -> Monthly Expiry (Last Tuesday)
        if len(expiry_str) == 5 and expiry_str[-3:].isalpha():
            dt = datetime.strptime(expiry_str, "%y%b")
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            dt_end = datetime(dt.year, dt.month, last_day)
            
            # Find the last Tuesday: weekday() returns 0 for Mon, 1 for Tue.
            offset = (dt_end.weekday() - 1) % 7 
            expiry_date = dt_end - timedelta(days=offset)
            return expiry_date.strftime("%Y-%m-%d 15:30:00")
            
        # Format: YYMDD (e.g., "26407", "26310", "26O07") -> Weekly Expiry
        elif len(expiry_str) == 5:
            year = int(expiry_str[:2]) + 2000
            m_char = expiry_str[2]
            
            # Handle NSE alphanumeric months (O=Oct, N=Nov, D=Dec)
            if m_char == 'O': month = 10
            elif m_char == 'N': month = 11
            elif m_char == 'D': month = 12
            else: month = int(m_char)
            
            day = int(expiry_str[3:])
            return datetime(year, month, day, 15, 30, 0).strftime("%Y-%m-%d 15:30:00")
            
        # Format fallback: YYMMDD (e.g., "260407")
        elif len(expiry_str) == 6 and expiry_str.isdigit():
            dt = datetime.strptime(expiry_str, "%y%m%d")
            return dt.strftime("%Y-%m-%d 15:30:00")
            
    except Exception:
        # Fail safe: return the original string so no data is lost if formatting is weird
        pass
        
    return expiry_str

def get_week_range_str(date_obj):
    """Calculates the Monday-to-Friday range for a given datetime to name the CSV."""
    start_of_week = date_obj - timedelta(days=date_obj.weekday()) 
    end_of_week = start_of_week + timedelta(days=4) 
    return f"{start_of_week.strftime('%Y-%m-%d')}_to_{end_of_week.strftime('%Y-%m-%d')}"

def process_and_save_csv(df, collection_name, data_dir):
    """Formats data cleanly, applies deduplication, and groups by week."""
    df['dt_obj'] = pd.to_datetime(df['datetime'])
    df['week_range'] = df['dt_obj'].apply(get_week_range_str)
    
    for week_str, group_df in df.groupby('week_range'):
        file_name = f"{collection_name}_{week_str}.csv"
        file_path = os.path.join(data_dir, file_name)
        
        # 1. Drop internal/temporary columns
        clean_group = group_df.drop(columns=['dt_obj', 'week_range', '_id'], errors='ignore').copy()
        clean_group.rename(columns={'volume': 'cumulative_volume'}, inplace=True)
        
        # 2. Standardize Floats to 2 decimal places
        for col in ['ltp', 'bid', 'ask']:
            if col in clean_group.columns:
                clean_group[col] = clean_group[col].astype(float).round(2)
                
        # 3. Parse Expiry to Unified IST Datetime String
        if 'expiry' in clean_group.columns:
            clean_group['expiry'] = clean_group['expiry'].apply(parse_expiry_to_str)
            
        # 4. Enforce strict unified column order
        desired_cols = ['symbol', 'type', 'expiry', 'strike', 'datetime', 'ltp', 'bid', 'ask', 'cumulative_volume']
        final_cols = [c for c in desired_cols if c in clean_group.columns]
        extra_cols = [c for c in clean_group.columns if c not in final_cols] # Keep unexpected cols safe
        clean_group = clean_group[final_cols + extra_cols]
        
        # 5. Save and Deduplicate
        if os.path.exists(file_path):
            logging.info(f"Updating existing file: {file_name}")
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, clean_group], ignore_index=True)
            combined_df.drop_duplicates(subset=['symbol', 'datetime'], keep='last', inplace=True)
            combined_df.sort_values(by='datetime', inplace=True)
            combined_df.to_csv(file_path, index=False)
        else:
            logging.info(f"Creating new file: {file_name}")
            clean_group.sort_values(by='datetime', inplace=True)
            clean_group.to_csv(file_path, index=False)

def backup_to_cache(db, cache_collection_name, docs):
    """Safely copies documents to a cache collection. Returns True only if
    every document is confirmed backed up; raises on quota/space errors
    instead of silently treating them as 'safe to proceed'."""
    if not docs:
        return True

    cache_collection = db[cache_collection_name]
    operations = [
        UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        for doc in docs
    ]

    try:
        result = cache_collection.bulk_write(operations, ordered=False)
        upserted = len(result.upserted_ids) if result.upserted_ids else 0
        matched = result.matched_count
        confirmed = upserted + matched
        if confirmed < len(docs):
            logging.warning(
                f"Only {confirmed}/{len(docs)} documents confirmed in "
                f"{cache_collection_name} — refusing to delete originals."
            )
            return False
        logging.info(f"Successfully cached {len(docs)} documents to {cache_collection_name}.")
        return True
    except BulkWriteError as bwe:
        write_errors = bwe.details.get('writeErrors', [])
        # Distinguish real duplicate-key conflicts (safe) from anything else (not safe)
        non_dup_errors = [e for e in write_errors if e.get('code') != 11000]
        if non_dup_errors:
            logging.error(
                f"Non-duplicate write errors while caching to {cache_collection_name}: "
                f"{non_dup_errors[:3]}"
            )
            return False
        n_ok = bwe.details.get('nUpserted', 0) + bwe.details.get('nMatched', 0)
        logging.warning(
            f"{n_ok}/{len(docs)} documents cached; remaining were duplicate-key conflicts. "
            "Safe to proceed."
        )
        return n_ok + len(non_dup_errors) >= len(docs) or n_ok == len(docs) - len(write_errors)
    except Exception as e:
        # Covers OperationFailure (e.g. Atlas quota-exceeded, code 8000) and anything else
        logging.error(f"Backup to {cache_collection_name} failed — writes are NOT confirmed: {e}")
        return False

def process_collection(db, collection_name, data_dir):
    """Core logic to fetch, save, cache, and prune a single collection."""
    logging.info(f"--- Processing Collection: {collection_name} ---")
    collection = db[collection_name]

    latest_global_doc = collection.find_one(sort=[("datetime", -1)])
    if not latest_global_doc:
        logging.info(f"Collection {collection_name} is empty. Skipping.")
        return

    cutoff_datetime = latest_global_doc['datetime']
    logging.info(f"Snapshot cutoff established at datetime: {cutoff_datetime}")

    docs = list(collection.find({"datetime": {"$lte": cutoff_datetime}}))

    latest_per_symbol = {}
    for doc in docs:
        sym = doc['symbol']
        if sym not in latest_per_symbol or doc['datetime'] > latest_per_symbol[sym]['datetime']:
            latest_per_symbol[sym] = doc

    ids_to_keep = {d['_id'] for d in latest_per_symbol.values()}

    if len(docs) <= len(ids_to_keep):
        logging.info("Only the latest documents for each symbol exist. Nothing to migrate.")
        return

    df = pd.DataFrame(docs)
    process_and_save_csv(df, collection_name, data_dir)
    logging.info(f"Local CSVs successfully written to {data_dir}.")

    docs_to_delete = [doc for doc in docs if doc['_id'] not in ids_to_keep]
    cache_col_name = f"cache_{collection_name}"

    # ── FIX: only delete from the source collection once the backup is
    # confirmed — otherwise we risk deleting data that was never archived. ──
    backup_confirmed = backup_to_cache(db, cache_col_name, docs_to_delete)
    if not backup_confirmed:
        logging.error(
            f"Backup to {cache_col_name} could not be confirmed — "
            f"ABORTING delete for {collection_name} this run to avoid data loss. "
            "Original records left in place; will retry next run."
        )
        return

    ids_to_delete = [doc['_id'] for doc in docs_to_delete]

    if ids_to_delete:
        try:
            delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
            logging.info(
                f"Deleted {delete_result.deleted_count} backed-up records from "
                f"{collection_name}. Retained {len(ids_to_keep)} latest symbol rows."
            )
            if delete_result.deleted_count < len(ids_to_delete):
                logging.warning(
                    f"Expected to delete {len(ids_to_delete)} but only "
                    f"{delete_result.deleted_count} were removed — check for concurrent writes."
                )
        except Exception as e:
            logging.error(
                f"Delete from {collection_name} FAILED — data was cached but NOT removed "
                f"from source, which will cause duplication/storage growth if not resolved: {e}"
            )
            raise

def run_fetch_pipeline():
    """Main execution function."""
    env_path, data_dir = resolve_paths()
    client = get_mongo_client(env_path)
    
    db = client['trading_data']
    target_collections = ['nifty_spreads']
    
    for coll in target_collections:
        try:
            process_collection(db, coll, data_dir)
        except Exception as e:
            logging.error(f"Critical error processing {coll}: {e}")
            
    client.close()
    logging.info("Pipeline execution complete.")

if __name__ == "__main__":
    run_fetch_pipeline()