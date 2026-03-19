#!/bin/bash

# Setup script for Kind cluster with GPU

echo "Installing Kind..."
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

echo "Creating cluster..."
kind create cluster --config kind-config.yaml

echo "Verifying cluster..."
kubectl get nodes

echo "Setting up GPU labels..."
kubectl label node kind-control-plane nvidia.com/gpu.present=true
kubectl label node kind-worker kind-worker2 kind-worker3 kind-worker4 nvidia.com/gpu.present=true

echo "Installing NVIDIA device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml

echo "Testing GPU on workers..."
docker cp /usr/bin/nvidia-smi kind-worker:/usr/bin/nvidia-smi
docker cp /usr/bin/nvidia-smi kind-worker2:/usr/bin/nvidia-smi
docker cp /usr/bin/nvidia-smi kind-worker3:/usr/bin/nvidia-smi
docker cp /usr/bin/nvidia-smi kind-worker4:/usr/bin/nvidia-smi

docker exec kind-worker nvidia-smi
docker exec kind-worker2 nvidia-smi
docker exec kind-worker3 nvidia-smi
docker exec kind-worker4 nvidia-smi

echo "Setup complete!"