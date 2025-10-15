
import random
from pymilvus import connections, Collection, utility

# --- Connection parameters ---
HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "book_collection"
DIM = 8  # Vector dimension
ID_TO_UPDATE = 3

def connect_to_milvus():
    """Connects to the Milvus server."""
    try:
        connections.connect("default", host=HOST, port=PORT)
        print(f"Successfully connected to Milvus at {HOST}:{PORT}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

def get_entity_before_update():
    """Retrieves and prints the entity before the update."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Run write.py first.")
        exit()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    expr = f"book_id == {ID_TO_UPDATE}"
    print(f"\nQuerying for entity before update ({expr})...")
    try:
        results = collection.query(expr=expr, output_fields=["book_id", "book_title", "book_embedding"])
        if not results:
            print(f"Entity with book_id {ID_TO_UPDATE} not found.")
        else:
            print("  Result before update:")
            for result in results:
                print(f"    - book_id: {result.get('book_id')}, book_title: {result.get('book_title')}, embedding_preview: {result.get('book_embedding')[:4]}...")
    except Exception as e:
        print(f"Failed to query entity before update: {e}")

def upsert_data():
    """Upserts data for a specific entity."""
    collection = Collection(COLLECTION_NAME)
    
    # Generate a new embedding and title for the book with ID_TO_UPDATE
    new_embedding = [0.5 for _ in range(DIM)] # A distinct, new vector
    new_title = f"Updated Book Title {ID_TO_UPDATE}"
    data = [
        [ID_TO_UPDATE],
        [new_title],
        [new_embedding]
    ]
    
    print(f"\nUpserting data for book_id {ID_TO_UPDATE}...")
    try:
        collection.upsert(data)
        collection.flush()
        print("Upsert successful and data flushed.")
    except Exception as e:
        print(f"Failed to upsert data: {e}")
        exit()

def get_entity_after_update():
    """Retrieves and prints the entity after the update to verify."""
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    expr = f"book_id == {ID_TO_UPDATE}"
    print(f"\nQuerying for entity after update ({expr})...")
    try:
        results = collection.query(expr=expr, output_fields=["book_id", "book_title", "book_embedding"])
        if not results:
            print(f"Entity with book_id {ID_TO_UPDATE} not found after upsert.")
        else:
            print("  Result after update:")
            for result in results:
                print(f"    - book_id: {result.get('book_id')}, book_title: {result.get('book_title')}, embedding_preview: {result.get('book_embedding')[:4]}...")
    except Exception as e:
        print(f"Failed to query entity after update: {e}")

if __name__ == "__main__":
    connect_to_milvus()
    get_entity_before_update()
    upsert_data()
    get_entity_after_update()
    print("\n--- Update script finished ---")
