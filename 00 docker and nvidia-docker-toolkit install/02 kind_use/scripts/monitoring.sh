#!/bin/bash

# Monitoring script for cluster status

echo "=== Cluster Status ==="
kubectl get nodes -o wide

echo -e "\n=== Pod Status ==="
kubectl get pods -A --no-headers | wc -l
kubectl get pods -A | grep -v Running

echo -e "\n=== GPU Resources ==="
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

echo -e "\n=== Exposed Ports ==="
docker ps | grep kind-control-plane | grep -o "127[0-9]*->"

echo -e "\n=== Disk Usage ==="
df -h | grep -E "(docker|overlay)"

echo "Monitoring complete."