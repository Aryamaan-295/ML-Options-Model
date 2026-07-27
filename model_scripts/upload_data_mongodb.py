import pymongo
import gridfs
import pandas as pd
import io
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# MongoDB connection URI from environment variable
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")

def push_csv_as_parquet_to_gridfs(mongo_uri, db_name, input_csv_path, target_filename):
    try:
        # 1. Verify local file exists before doing anything
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Local file '{input_csv_path}' does not exist.")

        # 2. Connect to MongoDB
        print("Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        
        # Initialize GridFS for the "processed_data" collection
        fs = gridfs.GridFS(db, collection="processed_data")
        
        # 3. Load local CSV and convert to Parquet in memory
        print(f"Reading local CSV '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path)
        
        print("Converting to Parquet binary format...")
        parquet_buffer = io.BytesIO()
        # pyarrow is required for this step
        df.to_parquet(parquet_buffer, engine='pyarrow', index=False) 
        parquet_bytes = parquet_buffer.getvalue()
        
        # 4. Find existing files to delete LATER (Safety first)
        print(f"Checking for existing versions of '{target_filename}'...")
        existing_files = list(fs.find({"filename": target_filename}))
        old_file_ids = [f._id for f in existing_files]
        
        if old_file_ids:
            print(f"Found {len(old_file_ids)} existing file(s). These will be replaced.")
        else:
            print("No existing files found. This will be a fresh upload.")

        # 5. Prepare Metadata (mimicking your original structure)
        # We extract rows, columns, and a timestamp to keep your DB schema consistent
        metadata = {
            "uploaded_at": datetime.utcnow().isoformat(),
            "rows": len(df),
            "columns": df.columns.tolist(),
            "source": "local-correction-upload"
        }
        
        # Optional: Try to get the last date if your 'Date' column exists
        if "Date" in df.columns:
            metadata["last_date"] = str(df["Date"].max())

        # 6. Upload the NEW file
        print(f"Uploading new file as '{target_filename}'...")
        # Using chunkSize=261120 to match your original database specifications exactly
        new_file_id = fs.put(
            parquet_bytes, 
            filename=target_filename, 
            metadata=metadata,
            chunkSize=261120 
        )
        print(f"Upload successful! New File ID: {new_file_id}")
        
        # 7. Delete the OLD files ONLY because the new upload succeeded
        if old_file_ids:
            print("Cleaning up old versions...")
            for old_id in old_file_ids:
                fs.delete(old_id)
                print(f"Deleted old file ID: {old_id}")
                
        print("Data update process completed successfully.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Process halted: {e}")
        print("No changes were made to the database. Your original data is safe.")
    finally:
        # Ensure the connection is closed
        if 'client' in locals():
            client.close()

# --- Execution ---
if __name__ == "__main__":
    DB_NAME = "deployments"           
    INPUT_CSV = "processed_data/nifty_master_file.csv"  # The local file you corrected
    TARGET_FILENAME = "nifty_master_file.parquet" # The name it should have in GridFS
    
    push_csv_as_parquet_to_gridfs(MONGO_URI, DB_NAME, INPUT_CSV, TARGET_FILENAME)