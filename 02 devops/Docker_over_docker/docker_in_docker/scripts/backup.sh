#!/bin/bash

# Automated Backup Script for Docker-in-Docker project
# Backs up volumes, configurations, and container data

set -e

BACKUP_DIR=${BACKUP_DIR:-/backups}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="dind_backup_$TIMESTAMP"

echo "💾 Starting backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup volumes
echo "📦 Backing up volumes..."
docker run --rm \
    -v dind_dind-data:/source \
    -v "$BACKUP_DIR/$BACKUP_NAME:/backup" \
    alpine tar czf "/backup/dind-data.tar.gz" -C /source .

docker run --rm \
    -v portainer_data:/source \
    -v "$BACKUP_DIR/$BACKUP_NAME:/backup" \
    alpine tar czf "/backup/portainer-data.tar.gz" -C /source .

# Backup configurations
echo "⚙️ Backing up configurations..."
cp -r /app/volumes/files "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
cp .env "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
cp docker-compose.yaml "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true

# Create backup manifest
cat > "$BACKUP_DIR/$BACKUP_NAME/manifest.txt" << EOF
Backup created: $(date)
Backup name: $BACKUP_NAME
Docker version: $(docker --version)
Compose version: $(docker-compose --version)
Included volumes: dind-data, portainer_data
Included configs: files, .env, docker-compose.yaml
EOF

# Compress backup
echo "🗜️ Compressing backup..."
cd "$BACKUP_DIR"
tar czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Clean up old backups
echo "🧹 Cleaning up old backups..."
find "$BACKUP_DIR" -name "dind_backup_*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete

echo "✅ Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
echo "📊 Backup size: $(du -sh "$BACKUP_DIR/$BACKUP_NAME.tar.gz" | cut -f1)"

# List current backups
echo "📋 Current backups:"
ls -la "$BACKUP_DIR"/dind_backup_*.tar.gz 2>/dev/null || echo "No backups found"