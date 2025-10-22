# install portainer:

docker run -d -p 9000:9000 --name portainer_test \
    -v /var/run/docker.sock:/var/run/docker.sock \
    portainer/portainer-ce


# install ssh
apk add --no-cache openssh-server nano which tmux
## change the default password
echo "root:password" | chpasswd
## change the default port
sed -i 's/#Port 22/Port 50422/' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

mkdir /run/sshd
ssh-keygen -A
chown root:root /var/empty
chmod 755 /var/empty




# new tmux session to run the ssh service there
tmux new -s ssh
# then:
/usr/sbin/sshd -D -p 50422





