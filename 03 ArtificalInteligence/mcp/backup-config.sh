#!/bin/bash

# Script to backup configuration files

set -e

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
CONFIG_FILES=(".env" "sample-config.json" "docker-compose.yml" "docker-compose.override.yml")

echo "📁 Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "💾 Backing up $file..."
        cp "$file" "$BACKUP_DIR/"
    else
        echo "⚠️  $file not found, skipping..."
    fi
done

# Create archive
ARCHIVE_NAME="mcp_config_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "📦 Creating archive: $ARCHIVE_NAME"
tar -czf "$ARCHIVE_NAME" -C "$BACKUP_DIR" .

echo "✅ Backup complete!"
echo "   Archive: $ARCHIVE_NAME"
echo "   Contents: ${CONFIG_FILES[*]}"

# Clean up temp directory
rm -rf "$BACKUP_DIR"