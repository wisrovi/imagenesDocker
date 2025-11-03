#!/bin/sh

# Get replica number from IP
TASK_SLOT=$(( $(hostname -I | awk '{print $1}' | cut -d. -f4) - 1 ))
hostname wisrovi-$TASK_SLOT

# Generate welcome message in /etc/motd
IP=$(ip addr show eth0 | grep inet | awk '{print $2}' | cut -d/ -f1)
WORKER_NUM=$(( $(echo $IP | cut -d. -f4) - 1 ))
cat > /etc/motd << EOF
$(figlet "Diun Worker")

=====================================

Worker Numero: $WORKER_NUM
IP del contenedor: $IP
Hostname: $(hostname)
Fecha: $(date)
Uptime: $(uptime | cut -d'up' -f2 | cut -d',' -f1)
Servicios: Docker $(docker --version | cut -d' ' -f3), Portainer en :9000, ttyd en :7681
Memoria: $(free -h | awk 'NR==2{print $3 "/" $2}') | CPU: $(nproc) cores
Comandos utiles: docker ps, htop, ping

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

# Start Portainer as container
mkdir -p /data/$TASK_SLOT
docker pull portainer/portainer
docker run -d --name portainer -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock -v /data/$TASK_SLOT:/data portainer/portainer --templates="" --no-analytics

# Start ttyd for web terminal
ttyd -p 7681 bash &

# Wait for all processes
wait