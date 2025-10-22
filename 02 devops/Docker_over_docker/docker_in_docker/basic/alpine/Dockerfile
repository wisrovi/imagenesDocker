# Basic Docker-in-Docker with SSH and Portainer
FROM docker:dind

# Install SSH, basic tools and ttyd for web terminal
RUN apk add --no-cache \
    openssh-server \
    tmux \
    curl \
    ttyd \
    && rm -rf /var/cache/apk/*

# Create SSH directory
RUN mkdir -p /run/sshd && \
    ssh-keygen -A && \
    echo "root:password" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#Port 22/Port 50422/' /etc/ssh/sshd_config

# Expose ports
EXPOSE 50422 9000 7681

# Copy start script
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/start.sh"]