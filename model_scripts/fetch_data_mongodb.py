import pymongo
import gridfs
import pandas as pd
import io
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# MongoDB connection URI from environment variable
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")

def extract_parquet_to_csv(mongo_uri, db_name, filename, output_csv_path):
    try:
        # 1. Connect to MongoDB
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        
        # 2. Initialize GridFS
        # By passing collection="processed_data", GridFS automatically looks 
        # for processed_data.files and processed_data.chunks
        fs = gridfs.GridFS(db, collection="processed_data")
        
        # 3. Find the file in the .files collection
        print(f"Searching for '{filename}' in GridFS...")
        file_data = fs.find_one({"filename": filename})
        
        if not file_data:
            raise FileNotFoundError(f"File '{filename}' not found in the database.")
            
        # 4. Read the binary chunks and reconstruct the file in memory
        print("Reconstructing binary data from chunks...")
        binary_parquet_data = file_data.read()
        
        # 5. Load the binary data into a Pandas DataFrame using a BytesIO buffer
        print("Loading Parquet data into Pandas...")
        parquet_buffer = io.BytesIO(binary_parquet_data)
        df = pd.read_parquet(parquet_buffer, engine='pyarrow')
        
        # 6. Save the DataFrame to a local CSV
        print(f"Saving data to '{output_csv_path}'...")
        df.to_csv(output_csv_path, index=False)
        
        print("Success! CSV file created.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the connection is closed
        if 'client' in locals():
            client.close()

# --- Execution ---
if __name__ == "__main__":
    # Replace these variables with your actual database details
    DB_NAME = "deployments"           # Replace with your actual DB name
    TARGET_FILENAME = "nifty_master_file.parquet"
    OUTPUT_CSV = "nifty_master_file_local.csv"
    
    extract_parquet_to_csv(MONGO_URI, DB_NAME, TARGET_FILENAME, OUTPUT_CSV)