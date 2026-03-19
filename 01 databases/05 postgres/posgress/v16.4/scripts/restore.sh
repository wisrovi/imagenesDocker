#!/bin/bash

# Restore script for PostgreSQL database in Docker container

CONTAINER_NAME="posgress_postgres_1"  # Adjust if your container name differs
BACKUP_FILE="$1"
DB_NAME="test_eyesnroad"
DB_USER="perseus"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql>"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Run pg_restore inside the container to restore (for custom format)
docker exec -i "$CONTAINER_NAME" pg_restore -U "$DB_USER" -d "$DB_NAME" -c "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Restore completed successfully from: $BACKUP_FILE"
else
    echo "Restore failed"
    exit 1
fi