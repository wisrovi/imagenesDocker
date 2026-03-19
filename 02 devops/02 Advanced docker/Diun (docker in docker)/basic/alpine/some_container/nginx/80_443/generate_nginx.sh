#!/bin/bash

SCALE=${SCALE:-4}

cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 80;
    server_name localhost;

    resolver 127.0.0.11;

EOF

for i in $(seq 1 $SCALE); do
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_pass http://172.19.0.$(($i + 2)):80/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

EOF
done

cat >> /etc/nginx/conf.d/default.conf <<EOF
    # Redirect to HTTPS
    return 301 https://\$server_name:50426\$request_uri;
}

server {
    listen 443 ssl;
    server_name localhost;

    ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;

    resolver 127.0.0.11;

EOF

for i in $(seq 1 $SCALE); do
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_pass http://172.19.0.$(($i + 2)):80/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

EOF
done

cat >> /etc/nginx/conf.d/default.conf <<EOF
}
EOF

exec nginx -g 'daemon off;'