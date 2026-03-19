#!/bin/sh

echo "Starting start.sh"

# Start Docker daemon in background
dockerd-entrypoint.sh &
DOCKER_PID=$!

echo "Docker daemon started with PID $DOCKER_PID"

# Wait for Docker to be ready
sleep 10

# Run installation scripts in parallel where possible
echo "Running installation scripts..."

# Start SSH installation in background
echo "Starting SSH installation..."
/usr/local/bin/install_ssh.sh &

# Start Portainer installation
echo "Running install_portainer.sh"
/usr/local/bin/install_portainer.sh

# Wait for SSH to complete
wait

# Setup SSL if enabled (can run in parallel with other services)
if [ "${SSL_ENABLED:-false}" = "true" ]; then
    echo "Running setup_ssl.sh"
    /usr/local/bin/setup_ssl.sh &
fi

# Start system services in parallel
echo "Starting system services..."

# Start log rotation
logrotate /etc/logrotate.d/app &

# Install cron jobs
echo "Installing cron jobs..."
crontab /etc/crontabs/crontab

# Start cron for scheduled tasks
crond -f -l 8 &

# Start Node Exporter for monitoring
if [ "${PROMETHEUS_ENABLED:-true}" = "true" ]; then
    /usr/local/bin/node_exporter --web.listen-address=:9100 &
fi

# Start fail2ban for security
echo "Starting fail2ban..."
fail2ban-server -f -s /var/run/fail2ban/fail2ban.sock &

# Wait for background SSL setup to complete
wait

echo "Scripts completed, waiting for Docker daemon"

# Wait for Docker daemon to keep container running
wait $DOCKER_PID