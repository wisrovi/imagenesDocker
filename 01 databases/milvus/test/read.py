

import random
from pymilvus import connections, Collection

# --- Connection parameters ---
HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "book_collection"
DIM = 8  # Vector dimension

def connect_to_milvus():
    """Connects to the Milvus server."""
    try:
        connections.connect("default", host=HOST, port=PORT)
        print(f"Successfully connected to Milvus at {HOST}:{PORT}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

def search_data():
    """Performs a vector similarity search."""
    try:
        collection = Collection(COLLECTION_NAME)
        # Load the collection into memory before searching
        print("Loading collection into memory...")
        collection.load()
        print("Collection loaded successfully.")

        # --- Prepare for search ---
        # Generate a random query vector
        query_vector = [random.random() for _ in range(DIM)]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16},
        }
        
        print("\nPerforming search...")
        # Search for the 2 most similar vectors
        results = collection.search(
            data=[query_vector],
            anns_field="book_embedding",
            param=search_params,
            limit=2,
            output_fields=["book_id", "book_title"]
        )
        
        print("Search results:")
        for i, hits in enumerate(results):
            print(f"  Query {i}:")
            for hit in hits:
                print(f"    - Hit: {hit}, book_id: {hit.entity.get('book_id')}, book_title: {hit.entity.get('book_title')}")

    except Exception as e:
        print(f"Failed to search data: {e}")
        exit()

if __name__ == "__main__":
    connect_to_milvus()
    search_data()
    print("\n--- Read script finished ---")

