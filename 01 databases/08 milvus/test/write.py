

import random
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

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

def create_collection():
    """Creates a collection in Milvus if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        return

    # --- Define Schema ---
    # Primary key field
    book_id = FieldSchema(
        name="book_id",
        dtype=DataType.INT64,
        is_primary=True,
        description="Primary key for the book",
    )
    # Vector field
    book_title = FieldSchema(
        name="book_title",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="Title of the book",
    )
    # Vector field
    book_embedding = FieldSchema(
        name="book_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=DIM,
        description="Vector embedding of the book",
    )
    # Schema definition
    schema = CollectionSchema(
        fields=[book_id, book_title, book_embedding],
        description="A collection to store book embeddings",
        enable_dynamic_field=False
    )

    try:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        collection = Collection(
            name=COLLECTION_NAME,
            schema=schema,
            using='default',
            consistency_level="Strong" 
        )
        print("Collection created successfully.")
    except Exception as e:
        print(f"Failed to create collection: {e}")
        exit()

def create_index():
    """Creates an index for the collection."""
    collection = Collection(COLLECTION_NAME)
    if collection.has_index():
        print("Index already exists. Skipping creation.")
        return

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    try:
        print("Creating index...")
        collection.create_index(
            field_name="book_embedding",
            index_params=index_params
        )
        print("Index created successfully.")
    except Exception as e:
        print(f"Failed to create index: {e}")
        exit()

def insert_data():
    """Inserts new, unique sample data into the collection."""
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # Get the current number of entities to generate unique IDs
    start_id = collection.num_entities
    num_new_records = 50
    end_id = start_id + num_new_records

    print(f"Collection has {start_id} entities. Inserting {num_new_records} new records...")
    
    # Generate 50 new random data points
    new_ids = [i for i in range(start_id, end_id)]
    new_embeddings = [[random.random() for _ in range(DIM)] for _ in range(num_new_records)]
    data = [
        new_ids,
        [f"Book Title {i}" for i in new_ids],
        new_embeddings,
    ]
    
    try:
        insert_result = collection.insert(data)
        # After insertion, it's important to flush the data to make it searchable
        collection.flush()
        print("Data inserted and flushed successfully.")
        # The num_entities property might not update immediately after flush, this is an expected behavior
        print(f"Expected number of entities now: {end_id}")
    except Exception as e:
        print(f"Failed to insert data: {e}")
        exit()

if __name__ == "__main__":
    connect_to_milvus()
    create_collection()
    create_index()
    insert_data()
    print("\n--- Write script finished ---")

