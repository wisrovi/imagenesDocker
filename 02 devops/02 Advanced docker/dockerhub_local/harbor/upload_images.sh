#!/bin/bash

docker login localhost:40232 -u user3 -p pass3Ghi789

images=("nginx" "redis" "mysql" "postgres" "mongo" "influxdb" "telegraf" "zookeeper" "cassandra" "couchdb" "memcached" "haproxy" "traefik" "httpd" "varnish" "squid" "postfix" "rabbitmq" "node" "hello-world" "alpine" "ubuntu" "debian" "busybox" "python" "golang" "openjdk" "tomcat" "jenkins" "wordpress")

for image in "${images[@]}"; do
    docker pull $image:latest
    docker tag $image:latest localhost:40232/user3-project/$image:latest
    docker push localhost:40232/user3-project/$image:latest
done