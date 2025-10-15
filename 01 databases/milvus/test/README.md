# Milvus Database Interaction Scripts

This project provides a set of Python scripts to perform common database operations on a [Milvus](https://milvus.io/) vector database. The scripts cover creating collections, inserting data, performing vector similarity searches, reading, updating, and deleting entities, and dropping collections.

## Author Information

- **Name**: WILLIAM R.
- **Title**: AI Leader & Solutions Architect at eCaptureDtech
- **Location**: BADAJOZ, EXTREMADURA, SPAIN
- **About**: As an AI Leader & Solutions Architect at eCaptureDtech, my mission is to bridge the gap between complex AI capabilities and real-world business challenges. I specialize in designing and implementing scalable, efficient, and innovative AI solutions that drive tangible results.

## Project Structure

The project is organized into several standalone scripts, each responsible for a specific task:

```
/
├─── delete.py           # Deletes a specific entity from the collection
├─── drop_collection.py  # Drops the entire collection
├─── read.py             # Performs a vector similarity search
├─── read_all.py         # Reads and displays all entities in the collection
├─── requirements.txt    # Project dependencies
├─── update.py           # Updates (upserts) a specific entity
└─── write.py            # Creates schema, collection, index, and inserts data
```

## Prerequisites

- Python 3.x
- A running Milvus instance. You can set one up easily using Docker. For more information, see the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).

## Setup

1.  **Clone the repository or download the scripts.**

2.  **Install the required Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the connection:**

    All scripts connect to a Milvus instance defined by the `HOST` and `PORT` variables at the top of each file. The default is `localhost:19530`. Modify these variables if your Milvus instance is running elsewhere.

    ```python
    # --- Connection parameters ---
    HOST = "localhost"
    PORT = "19530"
    COLLECTION_NAME = "book_collection"
    ```

## Usage

The scripts are designed to be run sequentially to observe the full workflow, but they can also be used independently.

### 1. `write.py`

This script is the starting point. It connects to Milvus and:
- Creates a collection named `book_collection` with a predefined schema if it doesn't already exist. The schema includes fields for `book_id` (primary key), `book_title`, and a float vector embedding (`book_embedding`).
- Creates an `IVF_FLAT` index on the vector field to optimize search performance.
- Inserts 50 sample records with unique IDs, titles, and randomly generated vector embeddings.

**To run:**
```bash
python write.py
```

### 2. `read_all.py`

This script fetches and displays all the records currently in the `book_collection`. It uses the `tabulate` library to present the data in a clean, readable table format.

**To run:**
```bash
python read_all.py
```

### 3. `read.py`

This script demonstrates the core functionality of a vector database: similarity search. It:
- Generates a random query vector.
- Searches the `book_collection` for the 2 most similar vectors based on the L2 distance metric.
- Prints the search results, including the ID and title of the matched books.

**To run:**
```bash
python read.py
```

### 4. `update.py`

This script shows how to update an existing entity using Milvus's `upsert` functionality. It targets a specific `book_id` (default is `3`) and:
- Queries the entity before the update to show its original state.
- Generates a new, distinct vector and title.
- Calls `upsert()` to overwrite the existing record with the new data.
- Queries the entity again to verify that the update was successful.

**To run:**
```bash
python update.py
```

### 5. `delete.py`

This script handles the deletion of a single entity. It targets a specific `book_id` (default is `0`) and:
- Deletes the entity using a query expression.
- Flushes the collection to ensure the deletion is committed.
- Verifies the deletion by attempting to query for the deleted entity.

**To run:**
```bash
python delete.py
```

### 6. `drop_collection.py`

This script is for cleanup. It completely removes the `book_collection` from the database.

**To run:**
```bash
python drop_collection.py
```
