#!/bin/bash

# SSL/TLS Setup Script for Docker-in-Docker project
# Sets up Let's Encrypt certificates for Portainer and documentation

set -e

echo "🔒 Setting up SSL/TLS certificates..."

# Load environment variables
DOMAIN_NAME=${DOMAIN_NAME:-localhost}
LETS_ENCRYPT_EMAIL=${EMAIL:-admin@example.com}
SSL_CERT_PATH=${SSL_CERT_PATH:-/etc/ssl/certs}
SSL_KEY_PATH=${SSL_KEY_PATH:-/etc/ssl/private}

# Create SSL directories
mkdir -p "$SSL_CERT_PATH" "$SSL_KEY_PATH"

# For localhost/development, generate self-signed certificates
if [ "$DOMAIN_NAME" = "localhost" ] || [ -z "$DOMAIN_NAME" ]; then
    echo "📝 Generating self-signed certificates for localhost..."

    # Generate private key
    openssl genrsa -out "$SSL_KEY_PATH/server.key" 2048

    # Generate certificate signing request
    cat > /tmp/cert.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
OU = Unit
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
EOF

    openssl req -new -x509 -key "$SSL_KEY_PATH/server.key" -out "$SSL_CERT_PATH/server.crt" -days 365 -config /tmp/cert.conf

    echo "✅ Self-signed certificates generated"
    echo "📍 Certificate: $SSL_CERT_PATH/server.crt"
    echo "🔑 Private Key: $SSL_KEY_PATH/server.key"

else
    echo "🌐 Setting up Let's Encrypt certificates for $DOMAIN_NAME..."

    # Install certbot if not available
    if ! command -v certbot &> /dev/null; then
        echo "Installing certbot..."
        apk add --no-cache certbot
    fi

    # Obtain certificate
    certbot certonly --standalone \
        --non-interactive \
        --agree-tos \
        --email "$LETS_ENCRYPT_EMAIL" \
        -d "$DOMAIN_NAME" \
        --cert-name "$DOMAIN_NAME"

    # Copy certificates to configured paths
    cp "/etc/letsencrypt/live/$DOMAIN_NAME/fullchain.pem" "$SSL_CERT_PATH/server.crt"
    cp "/etc/letsencrypt/live/$DOMAIN_NAME/privkey.pem" "$SSL_KEY_PATH/server.key"

    echo "✅ Let's Encrypt certificates obtained"
fi

# Set proper permissions
chmod 600 "$SSL_KEY_PATH/server.key"
chmod 644 "$SSL_CERT_PATH/server.crt"

# Configure firewall
echo "🔥 Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw --force enable
    ufw allow 50422/tcp  # SSH
    ufw allow 9000/tcp   # Portainer HTTP
    ufw allow 9443/tcp   # Portainer HTTPS
    ufw allow 80/tcp     # HTTP
    ufw allow 443/tcp    # HTTPS
    ufw allow 8082/tcp   # Docs
    ufw allow 9090/tcp   # Prometheus
    ufw allow 3000/tcp   # Grafana
    ufw allow 8080/tcp   # cAdvisor
    ufw allow 9100/tcp   # Node Exporter
    ufw --force reload
    echo "✅ Firewall configured"
else
    echo "⚠️ UFW not available, skipping firewall configuration"
fi

echo "🔒 SSL/TLS setup completed successfully!"