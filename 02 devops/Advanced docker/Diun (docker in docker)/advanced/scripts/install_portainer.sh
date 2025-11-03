#!/bin/sh

# Wait for Docker daemon to be ready
echo "Waiting for Docker daemon..."
timeout=60
count=0
while ! docker info > /dev/null 2>&1; do
  sleep 2
  count=$((count + 2))
  if [ $count -ge $timeout ]; then
    echo "Docker daemon did not start within $timeout seconds"
    exit 1
  fi
done
echo "Docker daemon ready"

# Install and start Portainer
echo "Starting Portainer..."
PORTAINER_ADMIN_USERNAME=${PORTAINER_ADMIN_USERNAME:-admin}
PORTAINER_ADMIN_PASSWORD=${PORTAINER_ADMIN_PASSWORD:-"Adm1nP@ssw0rd!"}

# Remove any existing portainer container
echo "Removing any existing portainer container..."
docker rm -f portainer 2>/dev/null && echo "Removed existing portainer container" || echo "No existing portainer container to remove"

# Create Portainer data volume if it doesn't exist
docker volume create portainer_data 2>/dev/null || true

docker run -d -p 9000:9000 --name portainer \
    --restart unless-stopped \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v portainer_data:/data \
    -e ADMIN_USERNAME="$PORTAINER_ADMIN_USERNAME" \
    -e ADMIN_PASSWORD="$PORTAINER_ADMIN_PASSWORD" \
    portainer/portainer-ce

# Wait for Portainer to be healthy
sleep 10
if docker ps | grep -q portainer; then
    echo "Portainer installed and started successfully on port 9000"
else
    echo "Failed to start Portainer"
    exit 1
fi