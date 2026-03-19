# Multi-stage build for optimized image
FROM alpine:latest AS base

# Install build dependencies
RUN apk add --no-cache \
    openssh-server \
    nano \
    which \
    tmux \
    ufw \
    fail2ban \
    certbot \
    openssl \
    curl \
    wget \
    htop \
    iotop \
    sysstat \
    logrotate \
    rsyslog \
    supervisor \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1001 -S dind && \
    adduser -u 1001 -S dind -G dind

FROM docker:dind AS final

# Copy packages from base stage
COPY --from=base /etc/apk/repositories /etc/apk/repositories
COPY --from=base /etc/apk/keys /etc/apk/keys
COPY --from=base /usr /usr
COPY --from=base /bin /bin
COPY --from=base /sbin /sbin
COPY --from=base /lib /lib
COPY --from=base /etc /etc
COPY --from=base /var /var

# Copy installation scripts
COPY scripts/install_ssh.sh /usr/local/bin/install_ssh.sh
COPY scripts/install_portainer.sh /usr/local/bin/install_portainer.sh
COPY scripts/setup_ssl.sh /usr/local/bin/setup_ssl.sh
COPY scripts/backup.sh /usr/local/bin/backup.sh

# Make scripts executable
RUN chmod +x /usr/local/bin/install_ssh.sh /usr/local/bin/install_portainer.sh /usr/local/bin/setup_ssl.sh /usr/local/bin/backup.sh

# Create necessary directories with proper permissions
RUN mkdir -p /etc/ssl/certs /etc/ssl/private /var/log/supervisor /etc/fail2ban/jail.d /backups && \
    chown -R root:root /etc/ssl /var/log/supervisor /etc/fail2ban /backups

# Configure log rotation
COPY config/logrotate.conf /etc/logrotate.d/app
RUN chmod 644 /etc/logrotate.d/app

# Configure fail2ban
COPY config/jail.local /etc/fail2ban/jail.d/custom.conf

# Configure cron jobs
COPY config/crontab /etc/crontabs/crontab

# Configure supervisor
COPY config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install Prometheus Node Exporter
RUN wget -q https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz && \
    tar xzf node_exporter-1.6.1.linux-amd64.tar.gz && \
    mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/ && \
    rm -rf node_exporter-1.6.1.linux-amd64*

# Expose ports (HTTP, HTTPS, SSH, Monitoring)
EXPOSE 80 443 9000 9001 50422 9100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD docker ps >/dev/null && curl -f http://localhost:9000/api/status >/dev/null 2>&1 && pgrep dockerd >/dev/null || exit 1

# Logging
VOLUME ["/var/log", "/backups"]

# Copy start script
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Set user and entrypoint
USER root
ENTRYPOINT ["/usr/local/bin/start.sh"]