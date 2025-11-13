# Kind Cluster with GPU Support

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.30+-blue.svg)](https://kubernetes.io/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-green.svg)](https://www.nvidia.com/)
[![Kind](https://img.shields.io/badge/Kind-v0.30.0-blue.svg)](https://kind.sigs.k8s.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive, production-ready setup for creating Kubernetes clusters using Kind (Kubernetes in Docker) with full NVIDIA GPU support, automated deployment scripts, and extensive documentation.

## 🚀 Overview

This project provides a complete, enterprise-grade solution for setting up lightweight Kubernetes clusters with GPU acceleration for development, testing, and CI/CD pipelines. It addresses the critical need for GPU-enabled Kubernetes environments in machine learning, AI, and high-performance computing workflows.

### 🎯 Key Capabilities

- **GPU Passthrough**: Direct access to NVIDIA GPUs via device mounts and library sharing
- **Multi-Node Architecture**: 1 control-plane + 4 worker nodes for distributed workloads
- **Port Exposure**: Pre-configured port mappings (12741-12761) for external service access
- **Automated Scripts**: One-command setup and comprehensive validation
- **Professional Documentation**: Multi-format documentation (Markdown, Sphinx, LaTeX) with examples
- **ArgoCD Integration**: Optional GitOps deployment support for continuous delivery
- **Monitoring & Observability**: Built-in health checks and performance monitoring
- **Security**: Proper device mounting and access controls

### 🏢 Use Cases

- **Machine Learning Development**: GPU-accelerated model training and inference
- **AI Research**: Distributed computing environments for data science
- **CI/CD Pipelines**: GPU-enabled testing and validation workflows
- **Edge Computing**: Lightweight Kubernetes for resource-constrained environments
- **Development Environments**: Consistent, reproducible Kubernetes setups across teams

## ✨ Key Features

- **GPU Support**: Full NVIDIA GPU passthrough with CUDA libraries and device plugin integration
- **High Availability**: Multi-worker node configuration for workload distribution and fault tolerance
- **Network Configuration**: Exposed ports (12741-12761) for service access and external connectivity
- **Security**: Proper device mounting, access controls, and container isolation
- **Monitoring**: Built-in validation, health checks, and performance monitoring scripts
- **Extensibility**: Easy integration with ArgoCD, monitoring tools, ingress controllers, and more
- **Automation**: Comprehensive setup scripts with error handling and rollback capabilities
- **Documentation**: Multi-format documentation with examples, troubleshooting guides, and API references

## 🏗️ Architecture

### Cluster Topology

```
┌─────────────────┐    ┌─────────────────┐
│   Control Plane │    │   Worker Node 1 │
│                 │    │  GPU: /dev/nvidia0 │
│ • API Server    │    │  Labels: gpu=true │
│ • Scheduler     │    │  Ports: exposed   │
│ • Controller    │    └─────────────────┘
│ • etcd          │
│ • Port Mapping  │    ┌─────────────────┐
│   (12741-12761) │    │   Worker Node 2 │
└─────────────────┘    │  GPU: /dev/nvidia0 │
                       │  Labels: gpu=true │
                       └─────────────────┘
                       ┌─────────────────┐
                       │   Worker Node 3 │
                       │  GPU: /dev/nvidia0 │
                       │  Labels: gpu=true │
                       └─────────────────┘
                       ┌─────────────────┐
                       │   Worker Node 4 │
                       │  GPU: /dev/nvidia0 │
                       │  Labels: gpu=true │
                       └─────────────────┘
```

- **Control Plane**: 1 node running Kubernetes control components (API server, scheduler, controller manager, etcd)
- **Worker Nodes**: 4 nodes optimized for GPU workloads with device passthrough
- **Networking**: Kindnet CNI with pod network 10.96.0.0/16 and service network 10.96.0.0/12
- **Storage**: HostPath-based persistent volumes with optional CSI driver support

### GPU Integration Architecture

```
Host System
├── NVIDIA Driver (575.64.03+)
├── CUDA Runtime (12.9+)
└── Device Files (/dev/nvidia*)
    ├── nvidia0 (GPU device)
    ├── nvidiactl (control device)
    ├── nvidiamodeset (modeset device)
    ├── nvidia-uvm (unified memory)
    └── nvidia-uvm-tools (UVMTools)

Container Runtime (Docker + NVIDIA Container Toolkit)
├── NVIDIA Container Runtime
├── GPU Device Mounts
└── Library Mounts
    ├── libnvidia-ml.so.1
    └── libcuda.so.1

Kubernetes Cluster (Kind)
├── Control Plane Node
│   └── GPU Labels: nvidia.com/gpu.present=true
├── Worker Nodes (x4)
│   ├── GPU Device Access
│   ├── NVIDIA Device Plugin
│   └── GPU Resource Scheduling
└── GPU Workloads
    ├── CUDA Applications
    ├── ML Training Jobs
    └── Inference Services
```

- **Device Passthrough**: Direct mounts of `/dev/nvidia*` devices for hardware access
- **Library Mounting**: CUDA and NVIDIA ML libraries shared from host to containers
- **Node Labeling**: Automatic `nvidia.com/gpu.present=true` labeling for GPU-aware scheduling
- **Device Plugin**: NVIDIA k8s-device-plugin for GPU resource management and scheduling
- **Runtime Configuration**: NVIDIA container runtime as default for GPU workloads

## 📋 Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| **NVIDIA GPU** | Any CUDA-compatible GPU | RTX 30-series or newer | RTX 3060, RTX 4070 |
| **RAM** | 8GB | 16GB+ | 32GB |
| **CPU Cores** | 4 cores | 8+ cores | 12 cores (AMD Ryzen/i7) |
| **Storage** | 50GB free | 100GB+ SSD | 500GB NVMe SSD |
| **Network** | 1Gbps | 10Gbps | 2.5Gbps |

### Software Prerequisites

#### Core Dependencies
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Docker**: 20.10+ (for Kind runtime and container management)
- **NVIDIA Drivers**: 575.64.03+ with CUDA 12.9+ compatibility
- **Linux Kernel**: 5.4+ (for GPU device passthrough support)

#### NVIDIA Software Stack
- **NVIDIA Driver**: Latest production driver from NVIDIA website
- **CUDA Toolkit**: 12.9+ (automatically installed with drivers)
- **NVIDIA Container Toolkit**: For Docker GPU support

#### Development Tools
- **curl/wget**: For downloading binaries and dependencies
- **git**: For cloning repositories and version control
- **make**: For build automation (optional)

### System Configuration

#### User Permissions
- **Sudo Access**: Required for system-wide installations and service management
- **Docker Group**: User must be in docker group or have sudo access to Docker commands
- **GPU Access**: User must have access to NVIDIA device files (`/dev/nvidia*`)

#### Network Requirements
- **Available Ports**: Localhost ports 12741-12761 must be free
- **Internet Access**: Required for downloading images and dependencies
- **DNS Resolution**: Functional DNS for container image pulls

#### Host System Validation
```bash
# Check NVIDIA GPU and drivers
nvidia-smi

# Verify Docker installation and GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check available ports
netstat -tln | grep -E '1274[1-9]|1275[0-9]|1276[01]'

# Verify sudo access
sudo -n true && echo "Sudo OK" || echo "Sudo requires password"
```

## 🛠️ Installation

### Quick Start (Recommended)

#### Prerequisites Validation
```bash
# Run comprehensive system validation
./scripts/validate.sh
```
This script checks:
- Docker installation and GPU support
- NVIDIA drivers and GPU availability
- System permissions and network connectivity
- Required tools and dependencies

#### One-Command Automated Setup
```bash
# Complete automated installation (requires sudo)
sudo ./scripts/mega_setup.sh
```
The mega setup script performs:
1. **System Validation**: Checks all prerequisites
2. **Docker Configuration**: Installs/configures Docker with NVIDIA runtime
3. **NVIDIA Toolkit**: Installs nvidia-container-toolkit
4. **Kubernetes Tools**: Installs Kind and kubectl
5. **Cluster Creation**: Creates Kind cluster with GPU support
6. **Device Plugin**: Installs NVIDIA k8s-device-plugin
7. **Validation**: Comprehensive testing of GPU functionality

### Manual Installation

#### Step 1: Install Docker and NVIDIA Container Toolkit
```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io

# Install NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit

# Configure Docker daemon for NVIDIA runtime
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

# Restart Docker
sudo systemctl restart docker
```

#### Step 2: Install Kubernetes Tools
```bash
# Install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/kubectl

# Verify installations
kind --version
kubectl version --client
```

#### Step 3: Create Kind Cluster
```bash
# Create cluster with GPU support
kind create cluster --config config/kind-config.yaml

# Install NVIDIA device plugin for GPU scheduling
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

### Alternative Installation Methods

#### Using Docker Compose
```bash
# Use docker-compose for isolated environment
docker-compose up -d
```

#### Using Makefile
```bash
# Use provided Makefile for automated builds
make install    # Full installation
make cluster    # Create cluster only
make test       # Run tests
make clean      # Cleanup
```

## 🚀 Usage

### Basic Cluster Operations

#### Creating the Cluster
```bash
# Create Kind cluster with GPU configuration
kind create cluster --config config/kind-config.yaml --name kind-gpu

# Wait for cluster to be ready (may take 2-5 minutes)
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

#### Verifying Cluster Health
```bash
# Check node status
kubectl get nodes -o wide

# Verify system pods
kubectl get pods -n kube-system

# Check cluster info
kubectl cluster-info
```

### GPU Operations

#### GPU Resource Validation
```bash
# Check GPU node labels
kubectl get nodes --show-labels | grep nvidia

# Describe GPU resources on nodes
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"

# Check NVIDIA device plugin status
kubectl get pods -n kube-system | grep nvidia-device-plugin
```

#### Running GPU Workloads
```bash
# Deploy GPU test pod
kubectl apply -f examples/gpu-pod-example.yaml

# Monitor pod status
kubectl get pods
kubectl logs gpu-pod

# Check GPU utilization
kubectl exec -it gpu-pod -- nvidia-smi
```

#### Advanced GPU Examples
```bash
# Multi-GPU workload
kubectl apply -f examples/multi-gpu-job.yaml

# GPU inference service
kubectl apply -f examples/gpu-inference-service.yaml

# ML training job
kubectl apply -f examples/ml-training-job.yaml
```

### Service Access and Networking

#### Port Mapping Reference
The cluster exposes services through port mappings on the control plane node:

| Service Port | Host Port | Common Use Case |
|--------------|-----------|-----------------|
| 80 | 12741 | HTTP web services |
| 443 | 12742 | HTTPS web services |
| 8080 | 12743 | Application dashboards |
| 3000-3009 | 12744-12753 | Development services |
| 5432 | 12754 | PostgreSQL databases |
| 6379 | 12755 | Redis cache |
| 9090 | 12756 | Prometheus monitoring |
| 3000-3100 | 12757-12761 | Additional services |

#### Accessing Services
```bash
# Example: Access web application on port 80
curl http://localhost:12741

# Port forward internal services
kubectl port-forward svc/my-service 8080:80

# Access Kubernetes dashboard (if installed)
kubectl port-forward svc/kubernetes-dashboard 8443:443 -n kubernetes-dashboard
open https://localhost:8443
```

### Development Workflows

#### Local Development Setup
```bash
# Set kubectl context
kubectl config use-context kind-kind-gpu

# Deploy development application
kubectl apply -f my-app/

# Enable port forwarding for hot reload
kubectl port-forward svc/my-app 3000:3000

# Monitor logs
kubectl logs -f deployment/my-app
```

#### CI/CD Integration
```bash
# Build and push images
docker build -t my-app:latest .
kind load docker-image my-app:latest

# Deploy to cluster
kubectl apply -f k8s/

# Run tests
kubectl run test-runner --image=my-app:latest --restart=Never --rm -it
```

#### Debugging and Monitoring
```bash
# Monitor cluster events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check pod resource usage
kubectl top pods

# Debug GPU workloads
kubectl describe pod gpu-pod
kubectl logs gpu-pod --previous
```

## ⚙️ Configuration

### Cluster Configuration Files

#### Kind Configuration (`config/kind-config.yaml`)
The main cluster configuration defines the complete Kind cluster topology:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  # Port mappings for external access
  extraPortMappings:
  - containerPort: 80
    hostPort: 12741
  # ... more port mappings
  # GPU device mounts
  extraMounts:
  - hostPath: /dev/nvidia0
    containerPath: /dev/nvidia0
  # ... more device mounts
  # Node labeling
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "nvidia.com/gpu.present=true"
- role: worker
  # Worker node configuration
  extraMounts:
  # GPU mounts for each worker
```

**Configuration Options:**
- **Node Count**: Control-plane (1) + Workers (4)
- **Port Range**: 12741-12761 for service exposure
- **GPU Devices**: All NVIDIA device files mounted
- **Libraries**: CUDA and ML libraries shared
- **Labels**: GPU presence labeling for scheduling

#### Kubeconfig (`config/kubeconfig.yaml`)
Generated automatically during cluster creation, contains:
- Cluster endpoint information
- Authentication credentials
- Context definitions
- User configurations

### GPU Configuration

#### Device Plugin Configuration
The NVIDIA k8s-device-plugin enables GPU scheduling:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.17.0
        name: nvidia-device-plugin-ctr
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
```

#### Runtime Class Configuration
For GPU workloads, use the NVIDIA runtime class:

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

### Advanced Configuration

#### Custom Resource Limits
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  runtimeClassName: nvidia
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
      requests:
        nvidia.com/gpu: 1
```

#### Node Affinity for GPU Workloads
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-workload
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpu-app
  template:
    metadata:
      labels:
        app: gpu-app
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.present
                operator: In
                values:
                - "true"
      containers:
      - name: gpu-container
        image: my-gpu-app:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### ArgoCD Integration

#### Installation
```bash
# Create ArgoCD namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd
```

#### Access ArgoCD UI
```bash
# Port forward ArgoCD server
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get initial admin password
kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath="{.data.password}" | base64 -d

# Open browser to https://localhost:8080
```

#### ArgoCD Application Example
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-gpu-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/my-org/my-gpu-app
    targetRevision: HEAD
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Monitoring and Observability

#### Prometheus and Grafana Setup
```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack

# Access Grafana
kubectl port-forward svc/monitoring-grafana 3000:80 -n default
# Default credentials: admin/prom-operator
```

#### GPU Monitoring
```bash
# Install DCGM (Data Center GPU Manager)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-monitoring-tools/main/dcgm-exporter.yaml

# Access metrics
kubectl port-forward svc/dcgm-exporter 9400:9400 -n kube-system
```

## 📚 Documentation

- **Sphinx Documentation**: [docs_sphinx/_build/html/index.html](docs_sphinx/_build/html/index.html)
  - Build: `cd docs_sphinx && make html`
  - Serve: `cd docs_sphinx && make serve`

- **Additional Resources**:
  - `docs/`: LaTeX documentation, architecture diagrams, troubleshooting guides
  - `docs/resources/`: Screenshots, command outputs, and reference materials

## 🔧 Troubleshooting

### Diagnostic Tools

#### Cluster Health Check
```bash
# Run comprehensive validation
./scripts/validate.sh

# Check cluster status
kubectl get nodes
kubectl get pods -n kube-system
kubectl cluster-info dump
```

#### GPU Diagnostics
```bash
# Host GPU check
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Container GPU access test
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Kubernetes GPU resources
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"
kubectl get pods -n kube-system | grep nvidia
```

#### Network Diagnostics
```bash
# Check port availability
netstat -tln | grep 1274[1-9]

# Test cluster networking
kubectl run test-pod --image=busybox --rm -it --restart=Never -- wget -O- http://kubernetes.default.svc.cluster.local

# Check CNI status
kubectl get pods -n kube-system | grep cni
kubectl logs -n kube-system kindnet-*
```

### Common Issues and Solutions

#### 1. Cluster Creation Failures

**Timeout During Creation**
```bash
# Increase timeout and reduce parallelism
kind create cluster --config config/kind-config.yaml --wait 10m --verbosity 1

# Check system resources
free -h
df -h
```

**Insufficient Resources**
```bash
# Reduce worker nodes in config
# Change from 4 workers to 2
sed -i 's/role: worker/role: worker/' config/kind-config.yaml
# Remove 2 worker node definitions
```

**Docker Issues**
```bash
# Check Docker status
sudo systemctl status docker
docker info

# Clean up Docker resources
docker system prune -a
docker volume prune
```

#### 2. GPU Access Problems

**GPU Not Detected in Containers**
```bash
# Verify NVIDIA Container Toolkit
nvidia-container-runtime --version

# Check Docker daemon configuration
cat /etc/docker/daemon.json

# Test GPU passthrough
docker run --rm --gpus all ubuntu nvidia-smi
```

**Device Plugin Issues**
```bash
# Check device plugin logs
kubectl logs -n kube-system daemonset/nvidia-device-plugin-daemonset

# Reinstall device plugin
kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

**Permission Denied on GPU Devices**
```bash
# Check device permissions
ls -la /dev/nvidia*

# Add user to video group
sudo usermod -aG video $USER

# Restart session or run:
newgrp video
```

#### 3. Networking Issues

**Port Conflicts**
```bash
# Find conflicting processes
lsof -i :12741-12761

# Kill conflicting processes
sudo kill -9 <PID>

# Change port mappings in config/kind-config.yaml
```

**Service Access Problems**
```bash
# Check service endpoints
kubectl get endpoints

# Test port forwarding
kubectl port-forward svc/my-service 8080:80 --address 0.0.0.0

# Verify firewall rules
sudo ufw status
sudo iptables -L
```

#### 4. Performance Issues

**Slow Cluster Startup**
```bash
# Use faster image registry
kind create cluster --config config/kind-config.yaml --image kindest/node:v1.30.0

# Pre-pull images
docker pull kindest/node:v1.30.0
```

**GPU Memory Issues**
```bash
# Monitor GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Check for memory leaks in pods
kubectl top pods --containers
```

**High CPU Usage**
```bash
# Check system processes
top -c

# Monitor Kubernetes components
kubectl top nodes
kubectl top pods -n kube-system
```

### Advanced Troubleshooting

#### Cluster Recovery
```bash
# Export cluster state
kind export logs ./kind-logs

# Recreate cluster
kind delete cluster
kind create cluster --config config/kind-config.yaml

# Restore workloads
kubectl apply -f my-workloads/
```

#### Debug Container Issues
```bash
# Debug pod with ephemeral container
kubectl debug my-pod --image=busybox --target=my-pod

# Check container logs with timestamps
kubectl logs my-pod --timestamps

# Execute into problematic container
kubectl exec -it my-pod -- /bin/bash
```

#### Network Debugging
```bash
# Install network debugging tools
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/netshoot.yaml

# Run network diagnostics
kubectl run netshoot --image=nicolaka/netshoot --rm -it --restart=Never
```

### Cleanup Procedures

#### Complete Cluster Removal
```bash
# Use cleanup script
./scripts/cleanup.sh

# Manual cleanup
kind delete cluster
docker system prune -a
docker volume prune
```

#### Selective Cleanup
```bash
# Remove specific resources
kubectl delete namespace my-namespace

# Clean up failed pods
kubectl delete pods --field-selector=status.phase=Failed

# Remove unused images
docker image prune -a
```

### Getting Help

#### Log Collection
```bash
# Collect all logs
./scripts/monitoring.sh

# Export cluster logs
kind export logs ./debug-logs

# System information
uname -a
lsb_release -a
docker --version
kubectl version
```

#### Community Support
- **GitHub Issues**: [Report bugs and request features](https://github.com/your-org/kind-gpu-cluster/issues)
- **Documentation**: Check `docs/` directory for detailed guides
- **Kind Community**: [Kind Slack](https://kind.sigs.k8s.io/#community)
- **NVIDIA Support**: [NVIDIA GPU Cloud](https://ngc.nvidia.com/)

### Prevention Best Practices

- **Regular Updates**: Keep NVIDIA drivers and CUDA toolkit updated
- **Resource Monitoring**: Monitor system resources during cluster operation
- **Backup Configurations**: Keep backups of working configurations
- **Test Environments**: Test configuration changes in development first
- **Documentation**: Document custom configurations and changes

## 📁 Project Structure

```
kind-gpu-cluster/
├── 📁 config/                          # Cluster configuration files
│   ├── kind-config.yaml               # Main Kind cluster configuration with GPU support
│   └── kubeconfig.yaml                # Generated Kubernetes configuration (created after cluster setup)
│
├── 📁 docs/                           # Documentation and resources
│   ├── 📁 resources/                  # Screenshots, diagrams, and reference materials
│   │   ├── 📁 results/               # Command outputs and test results
│   │   ├── argocd-*.png              # ArgoCD UI screenshots
│   │   └── architecture.png          # System architecture diagram
│   ├── architecture.md               # Detailed architecture documentation
│   ├── requirements.md               # Hardware and software requirements
│   ├── troubleshooting.md            # Comprehensive troubleshooting guide
│   ├── CHANGELOG.md                  # Version history and release notes
│   ├── FAQ.md                        # Frequently asked questions
│   ├── gpu_setup.txt                 # GPU configuration guide
│   ├── gpu_test.txt                  # GPU testing procedures
│   ├── install_kind.txt              # Kind installation instructions
│   ├── create_cluster.txt            # Cluster creation walkthrough
│   ├── verify_cluster.txt            # Cluster verification steps
│   ├── versions.txt                  # Version compatibility matrix
│   └── documentation.*               # LaTeX documentation (PDF, TEX, etc.)
│
├── 📁 docs_sphinx/                   # Sphinx documentation system
│   ├── 📁 _build/                   # Generated documentation (HTML, PDF)
│   │   └── 📁 html/                # Web documentation
│   ├── 📁 _static/                 # Static assets (CSS, JS, images)
│   ├── 📁 _templates/              # Sphinx templates
│   ├── conf.py                      # Sphinx configuration
│   ├── index.rst                    # Main documentation index
│   ├── installation.rst             # Installation guide
│   ├── configuration.rst            # Configuration reference
│   ├── usage.rst                    # Usage examples
│   ├── troubleshooting.rst          # Troubleshooting guide
│   ├── api_reference.rst            # API documentation
│   ├── examples.rst                 # Code examples
│   ├── bibliography.rst             # References and citations
│   ├── refs.bib                     # BibTeX bibliography
│   └── Makefile                     # Documentation build system
│
├── 📁 examples/                     # Example configurations and workloads
│   ├── gpu-pod-example.yaml         # Basic GPU pod example
│   ├── multi-gpu-job.yaml           # Multi-GPU job configuration
│   ├── gpu-inference-service.yaml   # GPU inference service
│   ├── ml-training-job.yaml         # Machine learning training job
│   └── tensorflow-gpu.yaml         # TensorFlow GPU workload
│
├── 📁 scripts/                      # Automation and utility scripts
│   ├── validate.sh                  # System prerequisites validation
│   ├── setup.sh                     # Basic setup script
│   ├── mega_setup.sh                # Complete automated setup (main installer)
│   ├── cleanup.sh                   # Cluster and resource cleanup
│   ├── monitoring.sh                # Health monitoring and diagnostics
│   ├── cluster_manager.py           # Python cluster management utilities
│   ├── gpu_monitor.py               # GPU monitoring and metrics
│   ├── set_path.sh                  # Environment path configuration
│   └── mega_setup_*.log             # Generated log files
│
├── 📄 argocd-app.yaml               # ArgoCD application configuration example
├── 📄 requirements.txt              # Python dependencies
├── 📄 Makefile                      # Build automation
├── 📄 LICENSE                       # MIT license
├── 📄 README.md                     # This file (comprehensive documentation)
└── 📄 .gitignore                    # Git ignore patterns
```

### File Descriptions

#### Configuration Files (`config/`)
- **`kind-config.yaml`**: Core cluster configuration defining node topology, GPU mounts, port mappings, and Kubernetes settings
- **`kubeconfig.yaml`**: Auto-generated Kubernetes client configuration for cluster access

#### Documentation (`docs/`)
- **Markdown Files**: Human-readable guides for architecture, requirements, and troubleshooting
- **Resources**: Visual aids, command outputs, and reference materials
- **LaTeX Documentation**: Professional PDF documentation with advanced formatting

#### Sphinx Documentation (`docs_sphinx/`)
- **Source Files (`.rst`)**: ReStructuredText documentation with cross-references and advanced formatting
- **Build System**: Automated generation of HTML and PDF documentation
- **Configuration**: Sphinx settings for themes, extensions, and output customization

#### Scripts (`scripts/`)
- **`validate.sh`**: Pre-flight checks for system compatibility
- **`mega_setup.sh`**: Comprehensive installation script with error handling
- **`cleanup.sh`**: Safe cluster removal and resource cleanup
- **`monitoring.sh`**: Health checks and performance monitoring
- **Python Scripts**: Advanced utilities for cluster management and GPU monitoring

#### Examples (`examples/`)
- **YAML Configurations**: Ready-to-deploy Kubernetes manifests
- **Workload Types**: GPU pods, jobs, services, and training workloads
- **Best Practices**: Production-ready examples with proper resource limits

#### Root Files
- **`Makefile`**: Automation for common tasks (install, test, clean, docs)
- **`requirements.txt`**: Python dependencies for management scripts
- **`argocd-app.yaml`**: Example ArgoCD application for GitOps deployments

## 📚 API Reference

### Configuration API

#### Kind Cluster Configuration Schema
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:                    # Array of node configurations
- role: control-plane     # Node role: control-plane or worker
  extraPortMappings:      # Port forwarding from host to container
  - containerPort: 80     # Port inside container
    hostPort: 12741       # Port on host system
    listenAddress: "127.0.0.1"  # Bind address (optional)
  extraMounts:           # Device and volume mounts
  - hostPath: /dev/nvidia0    # Path on host
    containerPath: /dev/nvidia0  # Path in container
  kubeadmConfigPatches:  # Kubernetes configuration patches
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "nvidia.com/gpu.present=true"
```

#### Port Mapping Reference
| Host Port | Container Port | Purpose |
|-----------|----------------|---------|
| 12741 | 80 | HTTP services |
| 12742 | 443 | HTTPS services |
| 12743 | 8080 | Application dashboards |
| 12744-12753 | 3000-3009 | Development services |
| 12754 | 5432 | PostgreSQL |
| 12755 | 6379 | Redis |
| 12756 | 9090 | Prometheus |
| 12757-12761 | 3100-3200 | Additional services |

### Script API

#### Mega Setup Script Options
```bash
sudo ./scripts/mega_setup.sh [options]

Options:
  -h, --help          Show help message
  -v, --verbose       Enable verbose output
  -s, --skip-validation Skip prerequisite checks
  -c, --cleanup-only  Only perform cleanup
  --no-gpu           Skip GPU-specific setup
```

#### Validation Script Exit Codes
- `0`: All checks passed
- `1`: Docker not available
- `2`: NVIDIA GPU/drivers not detected
- `3`: Insufficient permissions
- `4`: Network connectivity issues

### Kubernetes Resources

#### GPU Pod Specification
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  nodeSelector:           # Schedule on GPU nodes
    nvidia.com/gpu.present: "true"
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1    # Request 1 GPU
      requests:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
  restartPolicy: Never
```

#### GPU Job Specification
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-training-job
spec:
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
      - name: training
        image: my-ml-training:latest
        resources:
          limits:
            nvidia.com/gpu: 2
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
      restartPolicy: Never
```

### Environment Variables

#### Docker Configuration
```bash
# NVIDIA runtime configuration
export DOCKER_DEFAULT_RUNTIME=nvidia

# GPU device selection
export CUDA_VISIBLE_DEVICES=0,1

# Memory limits
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
```

#### Kubernetes Configuration
```bash
# Kubeconfig path
export KUBECONFIG=./config/kubeconfig.yaml

# Context selection
export KUBECTL_CONTEXT=kind-kind-gpu

# GPU resource requests
export GPU_REQUESTS=1
export GPU_LIMITS=1
```

## 🤝 Contributing

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/kind-gpu-cluster.git
   cd kind-gpu-cluster
   ```

2. **Setup Development Environment**
   ```bash
   ./scripts/validate.sh
   ./scripts/setup.sh
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Make Changes**
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation
   - Test on multiple GPU configurations

5. **Commit and Push**
   ```bash
   git add .
   git commit -m 'feat: add amazing GPU feature'
   git push origin feature/amazing-feature
   ```

6. **Create Pull Request**
   - Provide detailed description
   - Reference related issues
   - Include screenshots for UI changes

### Code Standards

#### Shell Scripts
- Use `set -euo pipefail` for error handling
- Include comprehensive error messages
- Add usage documentation
- Follow POSIX shell standards

#### Python Scripts
- Use type hints and docstrings
- Follow PEP 8 style guide
- Include unit tests
- Handle exceptions gracefully

#### Documentation
- Use clear, concise language
- Include code examples
- Keep screenshots updated
- Test all commands and examples

### Testing Guidelines

#### Automated Testing
```bash
# Run all tests
make test

# Test specific components
make test-gpu
make test-networking
make test-scripts
```

#### Manual Testing Checklist
- [ ] Cluster creation succeeds
- [ ] GPU detection works
- [ ] Port forwarding functional
- [ ] Device plugin installed
- [ ] Example workloads run
- [ ] Cleanup removes all resources

### Release Process

1. **Version Bump**: Update version in relevant files
2. **Changelog**: Document changes in `CHANGELOG.md`
3. **Testing**: Run full test suite
4. **Documentation**: Update Sphinx docs
5. **Tag Release**: Create Git tag
6. **Publish**: Push to repository

### Community Guidelines

- **Be Respectful**: Maintain professional communication
- **Help Others**: Share knowledge and assist newcomers
- **Report Issues**: Use issue templates for bug reports
- **Request Features**: Provide detailed use cases for feature requests
- **Code Reviews**: Provide constructive feedback

### Recognition

Contributors are recognized in:
- `CHANGELOG.md` for code contributions
- GitHub repository contributors list
- Release notes and acknowledgments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Permissions:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Limitations:**
- ❌ Liability
- ❌ Warranty

**Conditions:**
- © License and copyright notice

## 👤 Author & Maintainers

### Primary Author
**William R. Rodríguez**

*AI Leader & Solutions Architect at eCaptureDtech*

- **LinkedIn**: [wisrovi-rodriguez](https://es.linkedin.com/in/wisrovi-rodriguez)
- **Mission**: Forge the future of AI by connecting technology with people and business objectives
- **Expertise**: Kubernetes, GPU computing, MLOps, distributed systems

### Contributors
This project welcomes contributions from the community. See [Contributing](#-contributing) for details.

## 🙏 Acknowledgments

### Core Technologies
- **[Kind](https://kind.sigs.k8s.io/)** - Kubernetes in Docker for local development
- **[Kubernetes](https://kubernetes.io/)** - Container orchestration platform
- **[NVIDIA](https://www.nvidia.com/)** - GPU technology and CUDA ecosystem
- **[Docker](https://www.docker.com/)** - Container runtime and orchestration

### Supporting Tools
- **[ArgoCD](https://argo-cd.readthedocs.io/)** - GitOps continuous delivery
- **[Sphinx](https://www.sphinx-doc.org/)** - Documentation generation
- **[Prometheus](https://prometheus.io/)** - Monitoring and alerting
- **[Grafana](https://grafana.com/)** - Observability dashboards

### Community & Inspiration
- **CNCF Community** - Kubernetes ecosystem contributors
- **NVIDIA Developer Community** - GPU computing expertise
- **Open Source Community** - Inspiration and best practices

## 📊 Project Metrics

[![GitHub stars](https://img.shields.io/github/stars/your-org/kind-gpu-cluster?style=social)](https://github.com/your-org/kind-gpu-cluster/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/your-org/kind-gpu-cluster?style=social)](https://github.com/your-org/kind-gpu-cluster/fork)
[![GitHub issues](https://img.shields.io/github/issues/your-org/kind-gpu-cluster)](https://github.com/your-org/kind-gpu-cluster/issues)
[![GitHub PRs](https://img.shields.io/github/issues-pr/your-org/kind-gpu-cluster)](https://github.com/your-org/kind-gpu-cluster/pulls)

## 🔗 Related Projects

- **[kind](https://github.com/kubernetes-sigs/kind)** - Official Kind repository
- **[NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)** - GPU device plugin
- **[NVIDIA/gpu-monitoring-tools](https://github.com/NVIDIA/gpu-monitoring-tools)** - GPU monitoring
- **[argoproj/argo-cd](https://github.com/argoproj/argo-cd)** - ArgoCD GitOps

## 📞 Support & Contact

### Getting Help
- **📖 Documentation**: [docs_sphinx/_build/html/index.html](docs_sphinx/_build/html/index.html)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-org/kind-gpu-cluster/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-org/kind-gpu-cluster/discussions)
- **💬 Community**: [Kind Slack](https://kind.sigs.k8s.io/#community)

### Professional Services
For enterprise support, consulting, or custom development:
- **Email**: contact@ecapturedtech.com
- **Website**: [eCaptureDtech](https://ecapturedtech.com)
- **LinkedIn**: [William Rodríguez](https://es.linkedin.com/in/wisrovi-rodriguez)

## 🚀 Future Roadmap

### Planned Features
- [ ] Multi-GPU node configurations
- [ ] Automated GPU workload scaling
- [ ] Integration with ML frameworks (PyTorch, TensorFlow)
- [ ] Advanced monitoring dashboards
- [ ] CI/CD pipeline templates
- [ ] Cloud provider integrations
- [ ] Security hardening guides

### Version History
See [CHANGELOG.md](docs/CHANGELOG.md) for detailed version history.

---

<div align="center">

**Built with ❤️ for the open-source community**

*Empowering developers to harness GPU acceleration in Kubernetes*

[![Follow on LinkedIn](https://img.shields.io/badge/Follow%20on-LinkedIn-blue?style=flat&logo=linkedin)](https://es.linkedin.com/in/wisrovi-rodriguez)
[![Star on GitHub](https://img.shields.io/github/stars/your-org/kind-gpu-cluster?style=social)](https://github.com/your-org/kind-gpu-cluster)

</div>