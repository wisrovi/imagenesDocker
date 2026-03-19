
from pymilvus import connections, Collection, utility
from tabulate import tabulate

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

def read_all_data():
    """Reads all data from the collection and prints it in a table."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Run write.py first.")
        exit()

    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print(f"Collection has {collection.num_entities} entities.")

        # --- Query all entities ---
        # We query for a condition that is always true to get all results
        # Note: For very large datasets, this is not recommended. 
        # You should use iterators or paginated queries.
        print("\nQuerying all entities...")
        results = collection.query(
            expr="book_id >= 0",
            output_fields=["book_id", "book_title", "book_embedding"],
            limit=100 # Add a limit to avoid printing too much data
        )

        if not results:
            print("No entities found in the collection.")
            return

        # --- Format for tabulate ---
        table_data = []
        headers = ["Book ID", "Book Title", "Embedding (first 4 dims)"]
        for res in results:
            book_id = res["book_id"]
            book_title = res["book_title"]
            # Truncate the embedding for display purposes
            embedding_preview = str(res["book_embedding"][:4]) + "..."
            table_data.append([book_id, book_title, embedding_preview])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        print(f"Failed to read data: {e}")
        exit()

if __name__ == "__main__":
    connect_to_milvus()
    read_all_data()
    print("\n--- Read All script finished ---")
