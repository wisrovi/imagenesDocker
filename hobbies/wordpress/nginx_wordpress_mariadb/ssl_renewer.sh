#!/bin/bash
 
COMPOSE="/usr/local/bin/docker-compose –ansi never"
DOCKER="/usr/bin/docker"
 
cd /home/hackins/wordpress_docker/
$COMPOSE run certbot renew && $COMPOSE kill -s SIGHUP webserver
$DOCKER system prune -af

# ejecutar: crontab -e
# agregue al final del archivo:
# */5 * * * * /home/hackins/wordpress_docker/ssl_renewer.sh >> /var/log/cron_docker.log 2>&1

# luego:
# tail -f /var/log/cron_docker.log

# finalmente cambiar esa linea del crontab por:
# 0 18 * * * /home/hackins/wordpress_docker/ssl_renewer.sh >> /var/log/cron_docker.log 2>&1