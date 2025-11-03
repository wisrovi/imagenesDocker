#!/bin/bash

docker login localhost:40232 -u user2 -p pass2Def456

images=("nginx" "redis" "mysql" "postgres" "mongo" "influxdb" "telegraf" "zookeeper" "cassandra" "couchdb" "memcached" "haproxy" "traefik" "httpd" "varnish" "squid" "postfix" "rabbitmq" "node" "hello-world" "alpine" "ubuntu" "debian" "busybox" "python" "golang" "openjdk" "tomcat" "jenkins" "wordpress")

for img in "${images[@]}"; do
  docker pull $img:latest
  docker tag $img:latest localhost:40232/user2-project/$img:latest
  docker push localhost:40232/user2-project/$img:latest
done