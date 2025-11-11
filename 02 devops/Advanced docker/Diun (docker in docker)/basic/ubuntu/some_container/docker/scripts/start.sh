#!/bin/sh

echo "Starting Docker-in-Docker with SSH and Portainer..."

# Get replica number from IP
IP=$(ip addr show eth0 | grep inet | awk '{print $2}' | cut -d/ -f1)
TASK_SLOT=$(( $(echo $IP | cut -d. -f4) - 1 ))
hostname some_container-worker-$TASK_SLOT
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


# ------------------ INSTALACIÓN DE NVIDIA TOOLKIT --------------------------
echo "Installing NVIDIA Docker Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
echo "NVIDIA Docker Toolkit installed successfully."
# ------------------------------------------------------------------------------

# Clean up potential locks
rm -f /var/lib/docker/boltdb/bolt.db.lock || true

# Iniciar el servicio Docker
dockerd --host=unix:///var/run/docker.sock &
# dockerd --host=unix:///var/run/docker.sock --host tcp://0.0.0.0:2376 --tls=false --exec-opt native.cgroupdriver=cgroupfs &
echo "Docker daemon started"
# Wait for Docker to be ready
sleep 5
for i in {1..30}; do
  if docker info > /dev/null 2>&1; then
    break
  fi
  sleep 1
done





# Configure SSH password
SSH_PASSWORD=${SSH_PASSWORD:-password}
echo "root:$SSH_PASSWORD" | chpasswd

# Start SSH
echo "Starting SSH server..."
mkdir -p /run/sshd
/usr/sbin/sshd -D -p 50422 &
SSH_PID=$!

echo "SSH started on port 50422"
echo "Starting ttyd web terminal..."

# Start ttyd for web terminal
ttyd -p 7681 -i 0.0.0.0 bash &
echo "ttyd started on port 7681"


# Load images from shared volume
docker load < /data/images/portainer.tar || true
docker load < /data/images/filebrowser.tar || true
docker load < /data/images/nginx.tar || true
docker load < /data/images/nvidia.tar || true




# Portainer
docker rm -f portainer || true
docker run -d --privileged --name portainer \
    -p 9000:9000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/portainer_password.txt:/tmp/portainer_password.txt:ro \
    portainer/portainer-ce:latest \
    --admin-password-file /tmp/portainer_password.txt \
    --no-analytics

# File browser
mkdir -p ./folder_sharing ./config
docker rm -f filebrowser || true
docker run -d --privileged \
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
  hurlenko/filebrowser

sleep 1

# http
docker rm -f nginx80 nginx443 || true
docker run -d --privileged --name nginx80 -p 80:80 -v "/http/http_80/index.html:/usr/share/nginx/html/index.html:ro" --restart always nginx
docker run -d --privileged --name nginx443 -p 443:80 -v "/http/http_443/index.html:/usr/share/nginx/html/index.html:ro" --restart always nginx

sleep 4

# NVIDIA
docker rm -f NVIDIA || true
docker run -d --privileged --name NVIDIA --gpus all --health-cmd="nvidia-smi || exit 1" --health-interval=30s --health-retries=3 --health-timeout=5s nvidia/cuda:12.2.0-base-ubuntu22.04 bash -c "while true; do nvidia-smi || break; sleep 30; done; tail -f /dev/null"

docker run --gpus all --rm nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi


# Welcome
figlet "Welcome $hostname"

# Keep the container running
tail -f /dev/null
