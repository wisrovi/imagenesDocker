#!/bin/bash

SCALE=${SCALE:-5}



cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 9000;

EOF

for i in $(seq 1 $SCALE); do
    case $i in
        1) IP="172.19.0.4" ;;
        2) IP="172.19.0.5" ;;
        3) IP="172.19.0.6" ;;
        *) IP="some_container-worker-$i" ;;
    esac
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_redirect http://127.0.0.1:9000/ /;
        proxy_pass http://$IP:9000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto http;
    }

EOF
done

cat >> /etc/nginx/conf.d/default.conf <<EOF
}

server {
    listen 7681;

EOF

for i in $(seq 1 $SCALE); do
    case $i in
        1) IP="172.19.0.4" ;;
        2) IP="172.19.0.5" ;;
        3) IP="172.19.0.6" ;;
        *) IP="some_container-worker-$i" ;;
    esac
    cat >> /etc/nginx/conf.d/default.conf <<EOF
    location /worker$i/ {
        proxy_redirect http://127.0.0.1:7681/ /;
        proxy_pass http://$IP:7681/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto http;
    }

EOF
done

cat >> /etc/nginx/conf.d/default.conf <<EOF
}
EOF

nginx -t && exec nginx -g 'daemon off;'