#!/bin/bash

images=("nginx" "redis" "mysql" "postgres" "mongo" "elasticsearch" "kibana" "prometheus" "grafana" "influxdb" "telegraf" "consul" "vault" "etcd" "zookeeper" "cassandra" "couchdb" "memcached" "haproxy" "traefik" "apache" "varnish" "squid" "bind9" "postfix" "dovecot" "openldap" "389ds" "rabbitmq" "node")

for img in "${images[@]}"; do
  docker pull $img:latest
  docker tag $img:latest localhost:40232/user1-project/$img:latest
  docker push localhost:40232/user1-project/$img:latest
done