# instalar portainer:

docker run -d -p 9000:9000 --name portainer_test \
    -v /var/run/docker.sock:/var/run/docker.sock \
    portainer/portainer-ce


# instalar ssh
apk add --no-cache openssh-server nano which tmux
## cambio el password por default
echo "root:password" | chpasswd
## cambio el puerto por default
sed -i 's/#Port 22/Port 50422/' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

mkdir /run/sshd
ssh-keygen -A
chown root:root /var/empty
chmod 755 /var/empty




# nueva sesion de tmux para alli correr el servicio de ssh
tmux new -s ssh
# luego:
/usr/sbin/sshd -D -p 50422





