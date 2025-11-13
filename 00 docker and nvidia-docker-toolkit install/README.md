# AI Development Environment Setup

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.30+-blue.svg)](https://kubernetes.io/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-green.svg)](https://www.nvidia.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)

A comprehensive, production-ready setup for creating AI/ML development environments with Docker containerization and Kubernetes orchestration, featuring full NVIDIA GPU acceleration support.

## 🚀 Overview

This project provides a complete solution for setting up lightweight, GPU-accelerated development environments for artificial intelligence and machine learning workloads. It combines Docker containerization with Kubernetes orchestration using Kind (Kubernetes in Docker) to deliver:

- **GPU-Accelerated AI Development**: Full NVIDIA GPU support for TensorFlow, PyTorch, and other ML frameworks
- **Containerized Environments**: Reproducible, isolated development workspaces
- **Kubernetes Orchestration**: Multi-node cluster setup for distributed workloads
- **Automated Deployment**: One-command setup scripts for rapid environment provisioning
- **Professional Documentation**: Comprehensive guides, troubleshooting, and examples

The setup is divided into two main components:

1. **Docker Setup** (`docker/`): Base containerization with GPU support for AI development
2. **Kind Cluster** (`kind_use/`): Kubernetes orchestration with GPU passthrough for advanced deployments

## ✨ Key Features

### Docker Environment
- **GPU Integration**: NVIDIA Docker Toolkit with CUDA support
- **AI Frameworks**: Pre-configured TensorFlow, Jupyter, and development tools
- **Container Management**: Portainer UI for easy container orchestration
- **System Monitoring**: cAdvisor for resource monitoring and profiling
- **Development Tools**: VS Code, Chrome, Python packages, and utilities

### Kubernetes Cluster
- **Multi-Node Architecture**: 1 control-plane + 4 worker nodes
- **GPU Passthrough**: Direct device access with CUDA libraries
- **Port Exposure**: Pre-configured mappings for external service access
- **ArgoCD Integration**: Optional GitOps continuous delivery
- **Monitoring & Validation**: Built-in health checks and diagnostics

### Automation & Documentation
- **One-Command Setup**: Automated installation scripts
- **Comprehensive Documentation**: Sphinx and LaTeX guides with examples
- **Validation Scripts**: Prerequisites checking and health monitoring
- **Troubleshooting Guides**: Common issues and solutions
- **Professional Resources**: Architecture diagrams, screenshots, and reference materials

## 🏗️ Architecture

### Docker Setup Architecture
```
Host System (Ubuntu 18.04+)
├── NVIDIA GPU (CUDA-compatible)
├── Docker Engine + NVIDIA Toolkit
├── Development Containers
│   ├── TensorFlow GPU Jupyter
│   ├── Portainer (Management UI)
│   └── cAdvisor (Monitoring)
└── Development Tools
    ├── VS Code, Chrome
    ├── Python ML packages
    └── System utilities
```

### Kind Cluster Architecture
```
Kind Cluster (Kubernetes in Docker)
├── Control Plane Node
│   ├── Kubernetes API Server
│   ├── etcd
│   ├── Scheduler & Controller
│   └── Port Mappings (12741-12761)
├── Worker Nodes (x4)
│   ├── GPU Device Mounts
│   ├── CUDA Libraries
│   └── Application Workloads
└── Networking
    ├── Kindnet CNI
    ├── Pod Network: 10.96.0.0/16
    └── Service Exposure
```

### GPU Integration
- **Device Passthrough**: Direct mounts of `/dev/nvidia*` devices
- **Library Mounting**: CUDA runtime and NVIDIA ML libraries
- **Node Labeling**: Automatic GPU capability detection
- **Scheduling Support**: GPU-aware pod placement

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX/RTX series recommended)
- **RAM**: Minimum 8GB (16GB+ recommended for Kubernetes)
- **CPU**: 4 cores minimum (8+ cores recommended)
- **Storage**: 50GB+ free space for containers and data

### Software Prerequisites
- **Operating System**: Ubuntu 18.04+ (tested on 20.04, 22.04)
- **Docker**: 20.10+ with NVIDIA Docker Toolkit
- **NVIDIA Drivers**: 575.64.03+ with CUDA 12.9+
- **Linux Kernel**: 5.4+ for GPU passthrough
- **System Tools**: curl, wget, sudo access

### Network Requirements
- **Internet Connection**: Required for downloading dependencies
- **Port Availability**: Ports 12741-12761 for Kind services
- **Docker Group**: User must be in docker group or have sudo access

## 🛠️ Installation & Setup

### Quick Start (Recommended)

1. **Clone the Repository**:
   ```bash
   cd ~
   git clone <repository-url> ai-dev-setup
   cd ai-dev-setup
   ```

2. **Run Docker Setup**:
   ```bash
   cd docker
   chmod +x scripts/*.sh
   ./scripts/install_docker.sh
   ./scripts/install_tools.sh
   ./scripts/install_containers.sh
   ```

3. **Reboot System** (activate NVIDIA drivers)

4. **Run Kind Setup**:
   ```bash
   cd ../kind_use
   sudo ./scripts/mega_setup.sh
   ```

### Detailed Installation

#### Phase 1: Docker Environment Setup

Navigate to the `docker/` directory and follow the setup guide in `README_SETUP.md`:

```bash
# Install Docker and NVIDIA components
./scripts/install_docker.sh

# Install development tools
./scripts/install_tools.sh

# Setup containers and AI environment
./scripts/install_containers.sh
```

#### Phase 2: Kubernetes Cluster Setup

Navigate to the `kind_use/` directory:

```bash
# Validate prerequisites
./scripts/validate.sh

# Complete automated setup
sudo ./scripts/mega_setup.sh
```

### Verification

**Docker Environment**:
```bash
nvidia-smi                    # Check GPU drivers
docker --version             # Verify Docker installation
docker run --gpus all nvidia/cuda:12.9-base nvidia-smi  # Test GPU in container
```

**Kubernetes Cluster**:
```bash
kubectl get nodes           # Check cluster nodes
kubectl get pods -n kube-system  # Verify system pods
kubectl describe nodes | grep nvidia  # Check GPU labeling
```

## 🚀 Usage

### Docker Environment

#### Launch Jupyter with GPU
```bash
docker run -u $(id -u):$(id -g) --gpus all -d --name tf-jupyter \
  -v ~/notebooks:/tf/notebooks \
  -p 8888:8888 -p 6006:6006 \
  --user root -e GRANT_SUDO=yes \
  tensorflow/tensorflow:latest-gpu-jupyter
```
Access at: http://localhost:8888

#### Interactive TensorFlow Development
```bash
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu bash
```

#### Container Management
- **Portainer UI**: http://localhost:9000
- **System Monitoring**: http://localhost:8172 (cAdvisor)

### Kubernetes Cluster

#### Deploy GPU Workloads
```bash
# Apply GPU test pod
kubectl apply -f examples/gpu-pod-example.yaml

# Check GPU pod logs
kubectl logs gpu-test-pod
```

#### ArgoCD GitOps Setup
```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

#### Service Access
Services are exposed on localhost ports 12741-12761:
- Port 80 → localhost:12741
- Port 443 → localhost:12742

## 📚 Documentation

### Docker Setup Documentation
- `docker/README_SETUP.md`: Complete setup guide
- `docker/scripts/README.md`: Script documentation

### Kubernetes Documentation
- `kind_use/README.md`: Cluster setup and usage
- `kind_use/docs/architecture.md`: Technical architecture
- `kind_use/docs/requirements.md`: Detailed requirements
- `kind_use/docs/troubleshooting.md`: Common issues and solutions

### Professional Documentation
- **Sphinx Docs**: `kind_use/docs_sphinx/_build/html/index.html`
  ```bash
  cd kind_use/docs_sphinx
  make html  # Build documentation
  make serve # Serve locally
  ```

### Additional Resources
- `kind_use/docs/resources/`: Screenshots, command outputs, diagrams
- `kind_use/docs/FAQ.md`: Frequently asked questions
- `kind_use/docs/CHANGELOG.md`: Version history and updates

## 🔧 Troubleshooting

### Common Docker Issues

**GPU Not Detected in Containers**:
```bash
# Verify host GPU
nvidia-smi

# Check NVIDIA Docker installation
docker run --gpus all nvidia/cuda:12.9-base nvidia-smi

# Reinstall NVIDIA Docker toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Permission Denied**:
```bash
sudo usermod -aG docker $USER
# Logout and login again
```

### Common Kubernetes Issues

**Cluster Creation Timeout**:
```bash
# Reduce node count in config/kind-config.yaml
kind create cluster --config config/kind-config.yaml --wait 5m
```

**GPU Not Available in Pods**:
```bash
# Check node GPU labeling
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"

# Verify device mounts in kind config
ls -la /dev/nvidia*

# Check pod GPU requests
kubectl describe pod <gpu-pod-name>
```

**Nodes Not Ready**:
```bash
kubectl get pods -n kube-system
kubectl logs -n kube-system coredns-*
```

### Cleanup Procedures

**Remove Docker Environment**:
```bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker system prune -a
```

**Remove Kind Cluster**:
```bash
cd kind_use
./scripts/cleanup.sh
```

## 📁 Project Structure

```
ai-dev-setup/
├── docker/                          # Docker environment setup
│   ├── scripts/                     # Installation scripts
│   │   ├── install_docker.sh        # Docker + NVIDIA setup
│   │   ├── install_tools.sh         # Development tools
│   │   ├── install_containers.sh    # Container setup
│   │   └── README.md               # Script documentation
│   ├── docker-compose.yml           # cAdvisor monitoring
│   └── README_SETUP.md             # Docker setup guide
├── kind_use/                        # Kubernetes cluster setup
│   ├── config/                      # Kind configuration
│   │   ├── kind-config.yaml         # Cluster configuration
│   │   └── kubeconfig.yaml          # Generated kubeconfig
│   ├── docs/                        # Documentation and resources
│   │   ├── resources/               # Images, screenshots, results
│   │   ├── architecture.md          # Architecture overview
│   │   ├── requirements.md          # System requirements
│   │   ├── troubleshooting.md       # Troubleshooting guide
│   │   ├── FAQ.md                   # Frequently asked questions
│   │   ├── CHANGELOG.md             # Version history
│   │   └── *.txt                    # Command outputs and logs
│   ├── docs_sphinx/                 # Sphinx documentation
│   │   ├── _build/                  # Generated HTML docs
│   │   ├── conf.py                  # Sphinx configuration
│   │   └── *.rst                    # Documentation sources
│   ├── examples/                    # Example configurations
│   │   └── gpu-pod-example.yaml     # GPU workload example
│   ├── scripts/                     # Automation scripts
│   │   ├── validate.sh              # Prerequisites validation
│   │   ├── setup.sh                 # Basic setup
│   │   ├── mega_setup.sh            # Complete automated setup
│   │   ├── cleanup.sh               # Cluster cleanup
│   │   ├── monitoring.sh            # Health monitoring
│   │   ├── cluster_manager.py       # Python cluster management
│   │   └── gpu_monitor.py           # GPU monitoring script
│   ├── argocd-app.yaml              # ArgoCD application example
│   ├── requirements.txt             # Python dependencies
│   ├── Makefile                     # Build automation
│   ├── README.md                    # Kind setup guide
│   └── LICENSE                      # MIT License
├── README.md                        # This file (main project README)
└── .gitignore                       # Git ignore patterns
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make Changes**: Ensure code follows project conventions
4. **Test Thoroughly**: Run validation scripts and verify functionality
5. **Update Documentation**: Keep docs in sync with code changes
6. **Commit Changes**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
7. **Push to Branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request**

### Development Setup
```bash
git clone https://github.com/your-username/ai-dev-setup.git
cd ai-dev-setup

# Setup Docker environment
cd docker && ./scripts/install_docker.sh

# Setup Kind cluster
cd ../kind_use && ./scripts/validate.sh && ./scripts/setup.sh
```

### Code Standards
- Follow existing code style and conventions
- Add comprehensive documentation for new features
- Include tests for automation scripts
- Update README files for significant changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](kind_use/LICENSE) file for details.

## 👤 Author

**William R. Rodríguez**

*AI Leader & Solutions Architect at eCaptureDtech*

- LinkedIn: [https://es.linkedin.com/in/wisrovi-rodriguez](https://es.linkedin.com/in/wisrovi-rodriguez)
- Mission: Forge the future of AI by connecting technology with people and business objectives

## 🙏 Acknowledgments

### Core Technologies
- **[Docker](https://www.docker.com/)**: Containerization platform
- **[Kubernetes](https://kubernetes.io/)**: Container orchestration
- **[Kind](https://kind.sigs.k8s.io/)**: Kubernetes in Docker
- **[NVIDIA](https://www.nvidia.com/)**: GPU technology and CUDA
- **[TensorFlow](https://www.tensorflow.org/)**: Machine learning framework

### Tools & Libraries
- **ArgoCD**: GitOps continuous delivery
- **Portainer**: Docker container management
- **cAdvisor**: Container monitoring
- **Sphinx**: Documentation generation
- **LaTeX**: Professional documentation

### Community
Special thanks to the open-source community for the amazing tools and documentation that made this project possible.

---

*Built with ❤️ for the AI and DevOps community*