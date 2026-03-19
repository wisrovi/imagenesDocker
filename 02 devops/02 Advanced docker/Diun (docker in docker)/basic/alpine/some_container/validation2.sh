sshpass -p 'password' ssh worker1@192.168.1.84 -p 50422 'docker info'
sshpass -p 'password' ssh worker2@192.168.1.84 -p 50422 'docker info'
sshpass -p 'password' ssh worker3@192.168.1.84 -p 50422 'docker info'
sshpass -p 'password' ssh worker4@192.168.1.84 -p 50422 'docker info'
sshpass -p 'password' ssh worker5@192.168.1.84 -p 50422 'docker info'

curl -L -k http://192.168.1.84:50421/worker1
curl -L -k http://192.168.1.84:50421/worker2haz lo 
curl -L -k http://192.168.1.84:50421/worker3

curl -L -k http://192.168.1.84:50423/worker1
curl -L -k http://192.168.1.84:50423/worker2
curl -L -k http://192.168.1.84:50423/worker3