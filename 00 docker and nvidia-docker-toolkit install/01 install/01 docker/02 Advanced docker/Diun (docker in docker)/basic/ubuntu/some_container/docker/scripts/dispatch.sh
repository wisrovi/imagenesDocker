#!/bin/bash

replica_num=$(whoami | sed 's/worker//')

if [[ $replica_num =~ ^[0-9]+$ ]] && [ $replica_num -ge 1 ]; then

  IP="some_container-worker-$replica_num"
  if [ -z "$SSH_ORIGINAL_COMMAND" ]; then
    ssh -T -i /home/worker$replica_num/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$IP -p 50422
  else
    echo "Running: eval \"$SSH_ORIGINAL_COMMAND\"" >&2
    ssh -T -i /home/worker$replica_num/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$IP -p 50422 eval "$SSH_ORIGINAL_COMMAND"
  fi

else

  echo "Invalid replica: $USER"

  exit 1

fi