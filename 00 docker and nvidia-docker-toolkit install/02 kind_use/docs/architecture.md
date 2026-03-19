# Architecture

## Overview

This setup creates a Kind (Kubernetes in Docker) cluster with GPU support using device passthrough.

## Components

### Nodes
- **Control Plane**: 1 node running Kubernetes control components
- **Workers**: 4 nodes for application workloads

### Networking
- **CNI**: Kindnet (default)
- **Ports**: 12741-12761 exposed on control-plane for external access
- **Internal**: 10.96.0.0/16 pod network

### GPU Access
- **Method**: Device mounts + library mounts
- **Devices**: /dev/nvidia* (GPU, UVM, modeset, ctl)
- **Libraries**: libnvidia-ml.so.1, libcuda.so.1
- **Labels**: nvidia.com/gpu.present=true on all nodes

### Storage
- **CSI**: Standard storage class (hostPath)

## Security

- Root access in containers (Kind default)
- GPU device access via mounts
- No RBAC modifications

## Limitations

- Single control-plane for stability
- GPU scheduling via device plugin (optional)
- No persistent volumes by default