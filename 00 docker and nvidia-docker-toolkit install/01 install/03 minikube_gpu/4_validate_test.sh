#!/bin/bash

# Get the name of the pod created by the job
POD_NAME=$(kubectl get pods --selector=job-name=gpu-job -o=jsonpath='{.items[0].metadata.name}')

# Check if a pod name was found
if [ -z "$POD_NAME" ]; then
    echo "No pod found for job 'gpu-job'. Please check if the job was created successfully."
    exit 1
fi

# Get the logs of the pod
echo "Fetching logs for pod: $POD_NAME"
kubectl logs $POD_NAME
