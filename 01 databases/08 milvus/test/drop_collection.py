
from pymilvus import connections, utility

# --- Connection parameters ---
HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "book_collection"

def connect_to_milvus():
    """Connects to the Milvus server."""
    try:
        connections.connect("default", host=HOST, port=PORT)
        print(f"Successfully connected to Milvus at {HOST}:{PORT}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

def drop_collection():
    """Drops the collection if it exists."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping collection '{COLLECTION_NAME}'...")
        utility.drop_collection(COLLECTION_NAME)
        print("Collection dropped successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist. No need to drop.")

if __name__ == "__main__":
    connect_to_milvus()
    drop_collection()
    print("\n--- Drop Collection script finished ---")
