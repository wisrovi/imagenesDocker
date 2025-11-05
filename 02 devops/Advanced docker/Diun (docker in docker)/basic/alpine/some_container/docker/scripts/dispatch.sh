#!/bin/bash

replica_num=$(whoami | sed 's/worker//')

if [[ $replica_num =~ ^[0-9]+$ ]] && [ $replica_num -ge 1 ]; then

  IP="some_container-worker-$replica_num"
  if [ -z "$SSH_ORIGINAL_COMMAND" ]; then
    exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/$USER/.ssh/id_rsa -p 50422 root@$IP
  else
    exec ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/$USER/.ssh/id_rsa -p 50422 root@$IP sh -c "$SSH_ORIGINAL_COMMAND"
  fi

else

  echo "Invalid replica: $USER"

  exit 1

fi