FROM nginx:alpine

# Install certbot for SSL
RUN apk add --no-cache certbot certbot-nginx openssl

# Copy nginx configuration
COPY config/nginx.conf /etc/nginx/nginx.conf
COPY config/default.conf /etc/nginx/conf.d/default.conf

# Copy the built documentation
COPY _build/html /usr/share/nginx/html

# Create SSL directory
RUN mkdir -p /etc/ssl/certs /etc/ssl/private

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]