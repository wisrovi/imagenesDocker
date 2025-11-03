#!/bin/bash

# Script to update the MCP Inspector Docker image

set -e

echo "🔍 Checking for updates to MCP Inspector image..."

# Get current image digest
CURRENT_DIGEST=$(docker inspect ghcr.io/modelcontextprotocol/inspector:latest --format='{{index .RepoDigests 0}}' 2>/dev/null || echo "unknown")

echo "📥 Pulling latest image..."
docker pull ghcr.io/modelcontextprotocol/inspector:latest

# Get new image digest
NEW_DIGEST=$(docker inspect ghcr.io/modelcontextprotocol/inspector:latest --format='{{index .RepoDigests 0}}')

if [ "$CURRENT_DIGEST" != "$NEW_DIGEST" ]; then
    echo "✅ Image updated successfully!"
    echo "   Old: $CURRENT_DIGEST"
    echo "   New: $NEW_DIGEST"

    # Clean up old images
    echo "🧹 Cleaning up old images..."
    docker image prune -f

    echo "🎉 Update complete! Run 'make run' to use the new version."
else
    echo "ℹ️  Image is already up to date."
fi