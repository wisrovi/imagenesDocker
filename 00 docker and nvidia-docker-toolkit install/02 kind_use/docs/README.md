# Kind Cluster with GPU Support

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

A complete, professional setup for creating a Kubernetes cluster using Kind with NVIDIA GPU support, port exposure, and validation scripts.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Cluster Creation](#cluster-creation)
- [Verification](#verification)
- [GPU Setup](#gpu-setup)
- [Testing](#testing)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **Docker**: Installed and running (`docker info` works)
- **NVIDIA GPU**: With drivers installed (`nvidia-smi` available)
- **Linux Host**: With NVIDIA libraries in `/usr/lib/x86_64-linux-gnu/`
- **Permissions**: Sudo access for Kind installation

## Quick Start

```bash
# Clone or download this folder
cd kind_use

# Run automated setup
./scripts/setup.sh

# Verify
kubectl get nodes
```

## Installation

### Install Kind

Run the commands in `install_kind.txt`:

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
kind --version
```

### Install kubectl (if not present)

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

## Cluster Creation

Use `config/kind-config.yaml` to create the cluster:

```bash
kind create cluster --config config/kind-config.yaml
```

**What it creates:**
- 1 Control-plane node
- 4 Worker nodes
- Ports 12741-12761 exposed on control-plane
- GPU devices mounted on all nodes
- GPU presence labels

See `create_cluster.txt` for details.

## Verification

Check cluster status:

```bash
kubectl get nodes
kubectl get nodes --show-labels
kubectl cluster-info
```

All nodes should be `Ready` with GPU labels.

## GPU Setup

### Device Mounts

The config mounts:
- `/dev/nvidia*` devices
- NVIDIA libraries (`libnvidia-ml.so.1`, `libcuda.so.1`)

### Kubernetes GPU Scheduling

Install NVIDIA device plugin:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

Check GPU resources:

```bash
kubectl describe nodes | grep -A 5 nvidia.com/gpu
```

## Testing

### GPU Access Validation

Run commands from `gpu_test.txt`:

```bash
# Copy nvidia-smi to workers
docker cp /usr/bin/nvidia-smi kind-worker:/usr/bin/nvidia-smi
# ... for all workers

# Test
docker exec kind-worker nvidia-smi
```

Expected: GPU info from host.

### Automated Test

Use `setup.sh` which includes GPU testing.

## Usage Examples

### Run GPU Pod

```bash
kubectl apply -f examples/gpu-pod-example.yaml
kubectl logs gpu-pod
```

### Expose Service on Custom Port

Services can bind to exposed ports 12741-12761.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues:

- Cluster creation timeouts
- Nodes not ready
- GPU detection failures
- Port conflicts

## Cleanup

```bash
./scripts/cleanup.sh
# Or manually:
kind delete cluster
```

## Extras

### ArgoCD Installation

ArgoCD is a GitOps tool for Kubernetes.

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

Wait for pods to be ready:

```bash
kubectl get pods -n argocd
```

Access ArgoCD:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Open https://localhost:8080

Credentials:
- Username: admin
- Password: `kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d`

### k9s Installation

k9s is a terminal-based UI for Kubernetes.

```bash
curl -Lo ./k9s.tar.gz https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_amd64.tar.gz
tar -xzf k9s.tar.gz
chmod +x k9s
mkdir -p ~/bin
mv k9s ~/bin/
```

Add to PATH:

```bash
source scripts/set_path.sh
# Or permanently: echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
```

Run k9s:

```bash
k9s
```

It automatically uses the current kubeconfig context.

### Lens Integration

Import `config/kubeconfig.yaml` into Lens to visualize the cluster.

## Project Structure

```
kind_use/
├── config/
│   ├── kind-config.yaml      # Cluster configuration
│   └── kubeconfig.yaml       # Kubeconfig for external tools
├── docs/
│   ├── README.md             # This guide
│   ├── architecture.md       # System architecture
│   ├── requirements.md       # Prerequisites
│   ├── FAQ.md                # Frequently asked questions
│   ├── troubleshooting.md    # Troubleshooting guide
│   ├── CHANGELOG.md          # Change history
│   └── *.txt                 # Command reference files
├── examples/
│   └── gpu-pod-example.yaml  # Example GPU pod
├── scripts/
│   ├── setup.sh              # Automated setup
│   ├── cleanup.sh            # Cleanup script
│   ├── validate.sh           # Prerequisites check
│   ├── monitoring.sh         # Cluster monitoring
│   └── set_path.sh           # PATH setup
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT license
├── Makefile                  # Automation targets
└── versions.txt              # Version information
```

## Contributing

1. Fork and clone the repository
2. Create a feature branch
3. Make changes following the project structure
4. Test with `./scripts/setup.sh`
5. Update documentation if needed
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.