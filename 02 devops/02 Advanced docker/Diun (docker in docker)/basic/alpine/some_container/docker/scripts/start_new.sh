#!/bin/sh

# Get replica number from IP
IP=$(ip addr show eth0 | grep inet | awk '{print $2}' | cut -d/ -f1)
TASK_SLOT=$(( $(echo $IP | cut -d. -f4) - 1 ))
hostname wisrovi-$TASK_SLOT

# Generate welcome message in /etc/motd
IP=$(ip addr show eth0 | grep inet | awk '{print $2}' | cut -d/ -f1)
WORKER_NUM=$(( $(echo $IP | cut -d. -f4) - 1 ))
cat > /etc/motd << EOF
$(figlet "DinD Worker")

=====================================

Worker Numero: $WORKER_NUM
IP del contenedor: $IP
Hostname: $(hostname)
Fecha: $(date)
Uptime: $(uptime | cut -d'up' -f2 | cut -d',' -f1)
Servicios: Docker $(docker --version | cut -d' ' -f3), Portainer en :9000, ttyd en :7681
Memoria: $(free -h | awk 'NR==2{print $3 "/" $2}') | CPU: $(nproc) cores
Comandos utiles: dcu (docker-compose up -d), dps (docker ps), htop, ping, alias_finder (buscar alias, eg. alias_finder docker)

=====================================
creado por wisrovi, contactame al correo wisrovi.rodriguez@gmail.com

EOF

# Generate containerd config
containerd config default > /etc/containerd/config.toml

# Start containerd
containerd &

# Wait for containerd to be ready
while ! ctr version >/dev/null 2>&1; do
    sleep 1
done

# Start Docker daemon
dockerd --host unix:///var/run/docker.sock --host tcp://0.0.0.0:2376 --tls=false --exec-opt native.cgroupdriver=cgroupfs &

# Wait for Docker daemon to be ready
while ! docker info >/dev/null 2>&1; do
    sleep 1
done

# Start SSH
/usr/sbin/sshd -D &

# Start ttyd for web terminal
ttyd -p 7681 bash &

# Welcome
figlet "Welcome $hostname"



docker load < /docker_images/file_browser.tar
docker load < /docker_images/nginx.tar
docker load < /docker_images/portainer.tar




# Start Portainer as container
mkdir -p /data/$TASK_SLOT
echo -n "1234567891011" > /tmp/portainer_password.txt
docker run -d --name portainer \
    -p 9000:9000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/portainer_password.txt:/tmp/portainer_password.txt:ro \
    portainer/portainer-ce:latest \
    --admin-password-file /tmp/portainer_password.txt \
    --no-analytics 


# file browser
docker run -d \
    --name filebrowser \
    -p 4443:8080 \
    -v ./folder_sharing:/data \
    -v ./config:/config \
    -e PUID=1000 \
    -e PGID=1000 \
    -e FB_BASEURL=/ \
    -e VIRTUAL_HOST=wisrovi.com \
    -e VIRTUAL_PORT=8080 \
    -e LETSENCRYPT_HOST=wisrovi.com \
    -e LETSENCRYPT_EMAIL=wisrovi.rodriguez@gmail.com \
    --restart always \
    hurlenko/filebrowser:latest

docker run -d --name nginx80 -p 80:80 -v "/http/http_80/index.html:/usr/share/nginx/html/index.html:ro" --restart always nginx
docker run -d --name nginx443 -p 443:80 -v "/http/http_443/index.html:/usr/share/nginx/html/index.html:ro" --restart always nginx



# Wait for all processes
wait