#!/bin/bash

SCALE=${SCALE:-50}

# Array of IPs for workers 1-8, rest use container names
declare -a IPS=(
    [1]="172.19.0.8"  # worker1
    [2]="172.19.0.9"  # worker2
    [3]="172.19.0.3"  # worker3
    [4]="172.19.0.6"  # worker4
    [5]="172.19.0.5"  # worker5
    [6]="172.19.0.7"  # worker6
    [7]="172.19.0.4"  # worker7
    [8]="172.19.0.2"  # worker8
)

cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 9000;

EOF

for i in $(seq 1 $SCALE); do
    if [ $i -le 8 ] && [ "${IPS[$i]}" != "" ]; then
        IP="${IPS[$i]}"
    else
        IP="some_container-worker-$i"
    fi
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_redirect off;
        proxy_pass http://$IP:9000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

EOF
done

cat >> /etc/nginx/conf.d/default.conf <<EOF
}

server {
    listen 7681;

EOF

for i in $(seq 1 $SCALE); do
    if [ $i -le 8 ] && [ "${IPS[$i]}" != "" ]; then
        IP="${IPS[$i]}"
    else
        IP="some_container-worker-$i"
    fi
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_redirect off;
        proxy_pass http://$IP:7681/;
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

nginx -t && exec nginx -g 'daemon off;'