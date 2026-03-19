#!/bin/bash

# Script to create Kind cluster with GPU support and assign labels

# Delete existing cluster if any
kind delete cluster

# Create new cluster with config
kind create cluster --config config/kind-config.yaml --wait 10m

# Set kubeconfig
kind get kubeconfig > /tmp/kind-kubeconfig.yaml
export KUBECONFIG=/tmp/kind-kubeconfig.yaml

# Assign labels to workers
kubectl label node kind-worker worker=worker1 nvidia.com/gpu.present=true --overwrite
kubectl label node kind-worker2 worker=worker2 nvidia.com/gpu.present=true --overwrite
kubectl label node kind-worker3 worker=worker3 nvidia.com/gpu.present=true --overwrite
kubectl label node kind-worker4 worker=worker4 node-type=CPU --overwrite

# Extract kubeconfig for Lens
kind get kubeconfig > lens-kubeconfig.yaml

echo "Cluster created and labels assigned."
echo "Kubeconfig saved to lens-kubeconfig.yaml for use in Lens."