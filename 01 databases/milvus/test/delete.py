
from pymilvus import connections, Collection, utility

# --- Connection parameters ---
HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "book_collection"
ID_TO_DELETE = 0

def connect_to_milvus():
    """Connects to the Milvus server."""
    try:
        connections.connect("default", host=HOST, port=PORT)
        print(f"Successfully connected to Milvus at {HOST}:{PORT}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

def delete_entity():
    """Deletes a specific entity from the collection."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Run write.py first.")
        exit()

    collection = Collection(COLLECTION_NAME)
    collection.load()

    # Get number of entities before deletion
    print(f"Number of entities before deletion: {collection.num_entities}")

    # --- Expression to identify the entity to delete ---
    expr = f"book_id in [{ID_TO_DELETE}]"
    
    print(f"\nDeleting entity with expression: {expr}...")
    try:
        collection.delete(expr)
        collection.flush()
        print("Delete operation successful and data flushed.")
    except Exception as e:
        print(f"Failed to delete entity: {e}")
        exit()
    
    # Get number of entities after deletion
    print(f"Number of entities after deletion: {collection.num_entities}")

    # --- Verify deletion ---
    print(f"\nVerifying deletion by querying for book_id {ID_TO_DELETE}...")
    results = collection.query(expr=f"book_id == {ID_TO_DELETE}")
    if not results:
        print(f"Successfully verified. Entity with book_id {ID_TO_DELETE} not found.")
    else:
        print(f"Verification failed. Entity still exists: {results}")

if __name__ == "__main__":
    connect_to_milvus()
    delete_entity()
    print("\n--- Delete script finished ---")
