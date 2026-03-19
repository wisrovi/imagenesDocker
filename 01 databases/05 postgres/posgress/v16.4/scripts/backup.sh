#!/bin/bash

# Backup script for PostgreSQL database in Docker container

CONTAINER_NAME="posgress_postgres_1"  # Adjust if your container name differs
BACKUP_DIR="./backups"
DB_NAME="test_eyesnroad"
DB_USER="perseus"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate timestamp for backup file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_backup_$TIMESTAMP.sql"

# Run pg_dump inside the container (incremental-like with compression)
docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" --compress=9 --format=custom > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"

    # Rotate backups: keep last 7 daily backups
    cd "$BACKUP_DIR"
    ls -t ${DB_NAME}_backup_*.sql | tail -n +8 | xargs -r rm -f
    echo "Old backups rotated. Kept last 7."
else
    echo "Backup failed"
    exit 1
fi