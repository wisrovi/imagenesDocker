#!/bin/sh

# Install SSH and utilities
echo "Installing SSH server..."
apk add --no-cache openssh-server nano which tmux || { echo "Failed to install packages"; exit 1; }

# Change the default root password
echo "Setting root password..."
SSH_PASSWORD=${SSH_PASSWORD:-"Ch@ng3M3N0w!2024"}
echo "root:$SSH_PASSWORD" | chpasswd

# Configure SSH
echo "Configuring SSH..."
SSH_PORT=${SSH_PORT:-50422}
SSH_ALLOW_ROOT_LOGIN=${SSH_ALLOW_ROOT_LOGIN:-yes}
SSH_PASSWORD_AUTHENTICATION=${SSH_PASSWORD_AUTHENTICATION:-yes}

sed -i "s/#Port 22/Port $SSH_PORT/" /etc/ssh/sshd_config
sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin $SSH_ALLOW_ROOT_LOGIN/" /etc/ssh/sshd_config
sed -i "s/#PasswordAuthentication yes/PasswordAuthentication $SSH_PASSWORD_AUTHENTICATION/" /etc/ssh/sshd_config

# Additional security settings
sed -i 's/#MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
sed -i 's/#LoginGraceTime 2m/LoginGraceTime 30/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 300/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 2/' /etc/ssh/sshd_config

# Prepare SSH directories
mkdir -p /run/sshd
ssh-keygen -A > /dev/null 2>&1
chown root:root /var/empty
chmod 755 /var/empty

# Start SSH in tmux session
echo "Starting SSH server..."
tmux new -s ssh -d /usr/sbin/sshd -D -p 50422

# Verify SSH is running
sleep 2
if pgrep -f sshd > /dev/null; then
    echo "SSH installed and started successfully on port 50422"
else
    echo "Failed to start SSH"
    exit 1
fi