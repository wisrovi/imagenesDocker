# Milvus Docker Compose Setup

This project provides a comprehensive Docker Compose setup to run a full Milvus stack, including the Milvus vector database, its dependencies, and monitoring tools.

## Overview

This setup is designed for developers and data scientists who want to quickly get started with Milvus for building AI and vector search applications. It includes the following services:

- **Milvus Standalone:** The core vector database service.
- **etcd:** A distributed key-value store used by Milvus for metadata management.
- **MinIO:** A high-performance, S3-compatible object storage used by Milvus for data persistence.
- **Attu:** A modern and intuitive web-based GUI for Milvus.
- **Prometheus:** A powerful monitoring and alerting toolkit.
- **Grafana:** An open-source platform for monitoring and observability, pre-configured with a dashboard for Milvus.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

### 1. Clone the Repository

If you haven't already, clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>/docker
```

### 2. Start the Services

To start all the services in detached mode, run the following command from the directory containing the `docker-compose.yml` file:

```bash
docker-compose up -d
```

This command will pull the necessary Docker images and start all the containers in the background.

### 3. Verify the Services

You can check the status of the running containers using:

```bash
docker-compose ps
```

You should see all the services (`etcd`, `minio`, `standalone`, `attu`, `prometheus`, `grafana`) in the `Up` state.

### 4. Accessing the Services

Once the services are running, you can access them at the following URLs:

- **Milvus:** `localhost:19530`
- **Attu (Milvus GUI):** `http://localhost:8000`
- **MinIO Console:** `http://localhost:9100` (Console is on port 9002 as per compose file, but exposed on 9100)
- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000` (Default credentials: `admin`/`admin`)

## Basic Usage (Python Example)

Here is a simple Python script to demonstrate how to connect to Milvus, create a collection, insert vectors, and perform a search.

### 1. Install the Milvus Python SDK

First, you need to install the `pymilvus` library:

```bash
pip install pymilvus
```

### 2. Python Script

Create a Python file (e.g., `example.py`) and add the following code:

```python
import random
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# --- 1. Create a collection ---
collection_name = "my_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random_value", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]
schema = CollectionSchema(fields, "a simple collection")
collection = Collection(collection_name, schema)
print(f"Collection '{collection_name}' created.")

# --- 2. Insert data ---
print("Inserting data...")
entities = [
    [i for i in range(3000)],  # pk
    [random.random() for _ in range(3000)],  # random_value
    [[random.random() for _ in range(8)] for _ in range(3000)],  # embeddings
]
insert_result = collection.insert(entities)
collection.flush()
print(f"Data inserted. Num entities: {collection.num_entities}")

# --- 3. Create an index ---
print("Creating index...")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embeddings", index_params)
print("Index created.")

# --- 4. Load the collection and search ---
print("Loading collection...")
collection.load()

print("Searching...")
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = collection.search(
    data=[[random.random() for _ in range(8)]],
    anns_field="embeddings",
    param=search_params,
    limit=3,
    output_fields=["random_value"]
)

print("Search results:")
for hits in result:
    for hit in hits:
        print(f"  - Hit: {hit}, random_value: {hit.entity.get('random_value')}")

# --- 5. Release the collection ---
collection.release()
```

### 3. Run the script

Execute the script from your terminal:

```bash
python example.py
```

This script will connect to your local Milvus instance, create a collection, insert 3000 vectors, create an index, perform a similarity search, and print the results.

## Directory Structure

- `docker-compose.yml`: The main Docker Compose file that defines all the services.
- `volumes/`: This directory is used for data persistence.
  - `etcd/`: Stores etcd data.
  - `milvus/`: Stores Milvus data and metadata.
  - `minio/`: Stores data for MinIO.
- `monitoring/`: Contains configuration files for Prometheus and Grafana.

## Data Persistence

The data for Milvus, etcd, and MinIO is persisted in the `volumes` directory on the host machine. This ensures that your data is not lost when you stop and restart the services.

## Stopping the Services

To stop all the running services, use the following command:

```bash
docker-compose down
```

This will stop and remove the containers, but the data in the `volumes` directory will be preserved.

## Configuration

The `docker-compose.yml` file contains the configuration for all the services. You can modify this file to change ports, versions, or other settings.

### Environment Variables

The following environment variables are used in the `docker-compose.yml` file:

- `DOCKER_VOLUME_DIRECTORY`: The directory where the volumes are stored. Defaults to the current directory (`.`).
- `MINIO_ACCESS_KEY`: The access key for MinIO. Default: `minioadmin`.
- `MINIO_SECRET_KEY`: The secret key for MinIO. Default: `minioadmin`.

You can override these variables by creating a `.env` file in the same directory as the `docker-compose.yml` file.
