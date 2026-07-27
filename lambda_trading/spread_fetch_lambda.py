import time
import hashlib
import requests
import boto3
import csv
import re
from datetime import datetime, timezone, timedelta
from fyers_apiv3 import fyersModel
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# ==========================================
# CONFIGURATION
# ==========================================
IST = timezone(timedelta(hours=5, minutes=30))
UNDERLYING_BASE = "NIFTY"
AWS_REGION = "ap-south-1"
EVENTBRIDGE_RULE_NAMES = ["Fyers_Market_Hours_Schedule"]

ssm = boto3.client('ssm', region_name=AWS_REGION)
events_client = boto3.client('events', region_name=AWS_REGION)
scheduler_client = boto3.client('scheduler', region_name=AWS_REGION)

def get_secret(parameter_name):
    try:
        res = ssm.get_parameter(Name=f'/fyers/{parameter_name}', WithDecryption=True)
        return res['Parameter']['Value']
    except Exception as e:
        print(f"Failed to fetch {parameter_name}: {e}")
        return None

def send_telegram_alert(message):
    bot_token = get_secret('TELEGRAM_BOT_TOKEN')
    chat_id = get_secret('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("🚨 Telegram Alert Failed: Missing Bot Token or Chat ID.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message})
    except Exception as e:
        print(f"🚨 Telegram Network Error: {e}")

try:
    mongo_uri = get_secret('MONGO_URI')
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client['trading_data']
    spreads_collection = db['nifty_spreads']
    state_collection = db['tracking_state']
    universes_collection = db['universes']
    index_collection = db['nifty_5m_ohlcv']
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_symbol(symbol):
    try:
        parts = symbol.split(UNDERLYING_BASE)
        return parts[1][-2:], parts[1][:-2][:5], parts[1][:-2][5:] 
    except:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"

def get_rounded_5m_datetime(dt):
    """Rounds the given datetime down to the nearest 5-minute interval."""
    minute = dt.minute
    rounded_minute = minute - (minute % 5)
    return dt.replace(minute=rounded_minute, second=0, microsecond=0)

def fetch_options_universe(underlying_base):
    exchange_prefix = "BSE" if underlying_base in ["SENSEX", "BANKEX"] else "NSE"
    url = f"https://public.fyers.in/sym_details/{exchange_prefix}_FO.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        symbols = []
        pattern = re.compile(rf"^{exchange_prefix}:{underlying_base}\d+.*(CE|PE)$")
        for row in csv.reader(response.content.decode("utf-8").splitlines()):
            for field in row:
                if pattern.match(field.strip()):
                    symbols.append(field.strip())
                    break
        return symbols
    except Exception as e:
        print(f"🚨 Failed to fetch CSV: {e}") 
        return []

def generate_and_store_new_token():
    client_id = get_secret('APP_ID')
    secret_key = get_secret('SECRET_ID')
    refresh_token = get_secret('REFRESH_TOKEN')
    pin = get_secret('PIN')
    
    hash_str = hashlib.sha256(f"{client_id}:{secret_key}".encode()).hexdigest()
    url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
    
    response = requests.post(url, json={
        "grant_type": "refresh_token",
        "appIdHash": hash_str,
        "refresh_token": refresh_token,
        "pin": pin
    }).json()
    
    if response.get("s") == "ok":
        new_token = response["access_token"]
        ssm.put_parameter(Name='/fyers/ACCESS_TOKEN', Value=new_token, Type='SecureString', Overwrite=True)
        print("✅ New access token generated.")
        return new_token, client_id
    else:
        raise Exception(f"Failed to generate token: {response}")

def get_latest_closed_5m_candle(fyers, symbol="NSE:NIFTY50-INDEX"):
    """Fetches the most recently closed 5m OHLCV candle for the Index."""
    now_ist = datetime.now(IST)
    range_from = (now_ist - timedelta(days=3)).strftime("%Y-%m-%d")
    range_to = now_ist.strftime("%Y-%m-%d")
    
    data = {
        "symbol": symbol,
        "resolution": "5",
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }
    
    res = fyers.history(data=data)
    if res.get("s") != "ok":
        raise Exception(f"History API Error: {res}")
        
    candles = res.get("candles", [])
    if not candles:
        return None
        
    current_epoch = int(now_ist.timestamp())
    
    for candle in reversed(candles):
        candle_epoch = candle[0]
        # A 5m candle is fully closed 300 seconds after its start time
        if current_epoch >= candle_epoch + 300:
            return {
                "symbol": symbol,
                "datetime": datetime.fromtimestamp(candle_epoch, tz=IST).strftime("%Y-%m-%d %H:%M:%S"),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
                "timestamp": candle_epoch
            }
    return None

# ==========================================
# LAMBDA HANDLER
# ==========================================
def lambda_handler(event, context):
    current_run_time = datetime.now(IST)
    rounded_dt = get_rounded_5m_datetime(current_run_time)
    rounded_dt_str = rounded_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # We now track volume instead of tt
    state_doc = state_collection.find_one({"_id": "tracking_state"}) or {}
    last_volume_map = state_doc.get("volume_map", {})
    last_index_tt = state_doc.get("last_index_tt", 0)
    consecutive_failures = state_doc.get("failures", 0)
    
    alert_triggered = False 
    filtered_count = 0 
    msg = ""
    
    try:
        client_id = get_secret('APP_ID')
        access_token = get_secret('ACCESS_TOKEN')
        
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="/tmp/")
        
        test_req = fyers.quotes({"symbols": "NSE:NIFTY50-INDEX"})
        is_token_invalid = test_req.get("s") == "error" and test_req.get("code") in [-15, -300, -313]
        
        if is_token_invalid or "valid token" in str(test_req).lower() or "expired" in str(test_req).lower():
            print("Token expired. Generating new token...")
            access_token, client_id = generate_and_store_new_token()
            fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="/tmp/")
        
        # 1. Fetch and store the latest closed 5m Index Candle safely
        try:
            latest_candle = get_latest_closed_5m_candle(fyers)
            if latest_candle and latest_candle["timestamp"] > last_index_tt:
                index_collection.insert_one(latest_candle)
                last_index_tt = latest_candle["timestamp"]
                print(f"✅ Stored new Index Candle for {latest_candle['datetime']}")
        except Exception as idx_err:
            print(f"⚠️ Non-fatal error fetching Index OHLCV: {idx_err}")

        # 2. Handle the Options Universe
        today_str = current_run_time.strftime("%Y%m%d")
        universe_doc = universes_collection.find_one({"date": today_str})
        
        if not universe_doc:
            symbols_list = fetch_options_universe(UNDERLYING_BASE)
            symbols_list = [sym for sym in symbols_list if "INDEX" not in sym]
            if len(symbols_list) > 1:
                universes_collection.insert_one({"date": today_str, "symbols": symbols_list})
        else:
            symbols_list = universe_doc['symbols']
            symbols_list = [sym for sym in symbols_list if "INDEX" not in sym]
            
        symbol_chunks = [symbols_list[i:i + 50] for i in range(0, len(symbols_list), 50)]
        new_documents = []
        
        for chunk in symbol_chunks:
            time.sleep(0.15) 
            response = fyers.quotes(data={"symbols": ",".join(chunk)})
            
            if response.get("s") != "ok": 
                raise Exception(f"Quotes API Error: {response}")
            
            for item in response.get("d", []):
                symbol = item.get("n")
                v = item.get("v", {})
                volume = int(v.get("volume", 0))
                
                last_vol = last_volume_map.get(symbol, -1)
                
                # Deduplication logic based purely on volume
                if volume == 0 or volume == last_vol:
                    filtered_count += 1
                    continue
                
                # Volume has changed (or it's a new day and volume reset from previous close to a new >0 value)
                last_volume_map[symbol] = volume
                opt_type, expiry, strike = parse_symbol(symbol)
                
                new_documents.append({
                    "symbol": symbol,
                    "type": opt_type,
                    "expiry": expiry,
                    "strike": strike,
                    "datetime": rounded_dt_str, 
                    "ltp": v.get("lp", 0),
                    "bid": v.get("bid", 0),
                    "ask": v.get("ask", 0),
                    "volume": volume
                })
                
        if new_documents:
            try:
                spreads_collection.insert_many(new_documents, ordered=False) 
                consecutive_failures = 0  
                msg = f"Cycle complete. {len(new_documents)} options added at {rounded_dt_str}. {filtered_count} skipped (no vol change)."
                print(msg) 
            except BulkWriteError as bwe:
                # MongoDB caught duplicates and rejected them, but inserted the rest.
                inserted_count = bwe.details.get('nInserted', 0)
                duplicates_caught = len(new_documents) - inserted_count
                consecutive_failures = 0  # This is a successful failsafe, not a system failure
                msg = f"Cycle complete. {inserted_count} added, {duplicates_caught} duplicates blocked by DB. {filtered_count} skipped (no vol change)."
                print(msg)
        else:
            consecutive_failures = 0 
            msg = f"Cycle complete. 0 new options found. {filtered_count} skipped (no vol change)."
            print(msg)
            
    except Exception as e:
        consecutive_failures += 1
        msg = f"Critical Error encountered: {str(e)}"
        print(msg)
        alert_triggered = True 
    
    if alert_triggered:
        send_telegram_alert(f"⚠️ FYERS ALGO ERROR\n\n{msg}\n\nFailure Count: {consecutive_failures}/3\nTimestamp: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if consecutive_failures >= 3:
        send_telegram_alert("🚨 CIRCUIT BREAKER TRIGGERED\n\n3 consecutive failures. Disabling EventBridge schedule.")
        for rule_name in EVENTBRIDGE_RULE_NAMES:
            try:
                schedule = scheduler_client.get_schedule(Name=rule_name)
                scheduler_client.update_schedule(
                    Name=rule_name,
                    ScheduleExpression=schedule['ScheduleExpression'],
                    Target=schedule['Target'],
                    FlexibleTimeWindow=schedule['FlexibleTimeWindow'],
                    State='DISABLED'
                )
            except Exception as e:
                print(f"Could not disable schedule: {e}")
        consecutive_failures = 0
        
    state_collection.update_one(
        {"_id": "tracking_state"}, 
        {"$set": {
            "volume_map": last_volume_map, 
            "last_index_tt": last_index_tt,
            "failures": consecutive_failures
        }}, 
        upsert=True
    )
    
    return {"statusCode": 200 if not alert_triggered else 500, "body": msg}