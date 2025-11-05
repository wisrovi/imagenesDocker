#!/bin/sh

echo "Starting Docker-in-Docker with SSH and Portainer..."

# Start Docker daemon in background
dockerd-entrypoint.sh &
DOCKER_PID=$!

echo "Docker daemon started with PID $DOCKER_PID"

# Wait for Docker to be ready
sleep 5

# Configure SSH password
SSH_PASSWORD=${SSH_PASSWORD:-password}
echo "root:$SSH_PASSWORD" | chpasswd

# Start SSH
echo "Starting SSH server..."
/usr/sbin/sshd -D -p 50422 &
SSH_PID=$!

echo "SSH started on port 50422"

# Start ttyd for web terminal
echo "Starting ttyd web terminal..."
ttyd -W -p 7681 sh &
TTY_PID=$!

echo "ttyd started on port 7681"

# Wait a bit more for Docker
sleep 5

# Install and start Portainer
echo "Installing Portainer..."
docker run -d -p 9000:9000 --name portainer \
    --restart unless-stopped \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v portainer_data:/data \
    portainer/portainer-ce

echo "Portainer started on port 9000"
echo "Access Portainer at http://localhost:9000"
echo "SSH access: ssh root@localhost -p 50422 (password: password)"

# Wait for Docker daemon to keep container running
wait $DOCKER_PID