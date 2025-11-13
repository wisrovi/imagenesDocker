# Ubuntu AI Docker GPU Setup Guide

## Overview

This comprehensive guide provides step-by-step instructions to transform a fresh Ubuntu installation into a fully configured environment optimized for AI and machine learning development. The setup integrates Docker containerization with NVIDIA GPU acceleration, enabling efficient and reproducible workflows for TensorFlow-based projects.

The project automates the installation of essential components including Docker, NVIDIA drivers, development tools, and pre-configured containers for AI workloads. By following this guide, users can quickly establish a professional-grade development environment without manual configuration hassles.

## Prerequisites

Before proceeding, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 18.04 or later (Ubuntu 20.04 or 22.04 recommended for best compatibility)
- **Hardware**: NVIDIA GPU with CUDA support (GTX/RTX series preferred; minimum GTX 1050 or equivalent)
- **System Resources**: At least 8GB RAM, 50GB free disk space
- **Network**: Stable internet connection for downloading packages and Docker images
- **Permissions**: Administrator (sudo) privileges
- **BIOS Settings**: Ensure virtualization is enabled in BIOS/UEFI

**Note**: This setup is specifically designed for NVIDIA GPUs. AMD GPU users should explore alternative containerization solutions.

## Project Structure

```
.
├── scripts/
│   ├── install_docker.sh      # Docker, NVIDIA drivers, and toolkit installation
│   ├── install_tools.sh       # Development tools and utilities
│   ├── install_containers.sh  # Container setup and AI environment
│   └── README.md              # Scripts documentation
├── docker-compose.yml         # cAdvisor monitoring configuration
└── README_SETUP.md            # This setup guide
```

## Quick Start (Automated Installation)

For most users, the automated installation provides the fastest path to a working environment:

### Step 1: Obtain the Project Files

Clone or download this repository to your home directory:

```bash
cd ~
git clone <repository-url> ubuntu-ai-docker-setup
cd ubuntu-ai-docker-setup
```

Replace `<repository-url>` with the actual repository URL.

### Step 2: Prepare and Execute Installation Scripts

Make all scripts executable and run them in sequence:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Install Docker, NVIDIA drivers, and GPU toolkit
./scripts/install_docker.sh

# Install development tools and utilities
./scripts/install_tools.sh

# Set up containers and AI environment
./scripts/install_containers.sh
```

**Important**: The `install_docker.sh` script will automatically reboot your system after installing NVIDIA drivers. Resume the installation from `install_tools.sh` after reboot.

### Step 3: Verify Installation

After completion, verify all components are working correctly:

```bash
# Check NVIDIA GPU and drivers
nvidia-smi

# Verify Docker installation
docker --version
docker run hello-world

# Check Docker Compose
docker-compose --version

# Test GPU integration with Docker
docker run --gpus all --rm nvidia/cuda:11.0-base nvidia-smi
```

## Detailed Installation (Manual Setup)

If the automated scripts encounter issues or you prefer manual control, follow these detailed steps:

### Phase 1: Docker and NVIDIA Setup

Execute the Docker installation script:

```bash
./scripts/install_docker.sh
```

This script performs the following actions:

1. **NVIDIA Driver Installation**:
   - Detects your GPU model
   - Downloads and installs appropriate proprietary drivers
   - Configures kernel modules for GPU access

2. **Docker Engine Installation**:
   - Adds Docker's official GPG key and repository
   - Installs Docker CE (Community Edition)
   - Adds your user to the `docker` group for non-root access

3. **Docker Compose Installation**:
   - Downloads the latest stable release
   - Installs the binary to `/usr/local/bin/`

4. **NVIDIA Container Toolkit**:
   - Adds NVIDIA's Docker repository
   - Installs `nvidia-docker2` or `nvidia-container-toolkit`
   - Enables GPU passthrough in containers

**System Reboot Required**: After driver installation, reboot to activate changes.

### Phase 2: Development Tools Installation

Run the tools installation script:

```bash
./scripts/install_tools.sh
```

This installs a comprehensive set of development tools:

- **Code Editors**: Visual Studio Code, Notepad++
- **Web Browsers**: Google Chrome
- **Remote Access**: AnyDesk for remote desktop connections
- **Python Environment**: Python 3, pip, essential libraries (tqdm, selenium)
- **System Utilities**: Stacer (system monitor), BleachBit (cleaner), Timeshift (backup)
- **Docker Tools**: Lazydocker (terminal UI for Docker management)

### Phase 3: Container and AI Environment Setup

Execute the container setup script:

```bash
./scripts/install_containers.sh
```

This script configures the AI development environment:

1. **Portainer Installation**:
   - Deploys Portainer CE for web-based container management
   - Accessible at http://localhost:9000

2. **GPU Integration Testing**:
   - Runs NVIDIA CUDA container to verify GPU access
   - Creates a persistent test container for ongoing validation

3. **TensorFlow Images**:
   - Pulls official TensorFlow GPU images
   - Includes both standard and Jupyter variants

4. **Jupyter Notebook Environment**:
   - Launches GPU-accelerated Jupyter server
   - Mounts current directory for notebook access
   - Accessible at http://localhost:8888

## Using the AI Development Environment

### Starting Jupyter Notebook

The environment includes a pre-configured Jupyter server with GPU support:

```bash
# The container is already running after installation
# Access via web browser at http://localhost:8888
```

For custom configurations:

```bash
docker run -u $(id -u):$(id -g) --gpus all -d --name tf-jupyter \
  -v ~/notebooks:/tf/notebooks \
  -v ~/datasets:/tf/datasets \
  -p 8888:8888 -p 6006:6006 \
  --user root -e GRANT_SUDO=yes \
  tensorflow/tensorflow:latest-gpu-jupyter
```

### Interactive TensorFlow Development

For command-line development or debugging:

```bash
# Start interactive TensorFlow container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  tensorflow/tensorflow:latest-gpu \
  bash
```

### TensorBoard Visualization

Monitor training progress with TensorBoard:

```bash
# TensorBoard is accessible at http://localhost:6006
# Logs are automatically available when running TensorFlow in containers
```

## Management and Monitoring Tools

### Portainer (Container Management)

Access the web-based Docker management interface at http://localhost:9000 to:
- View running containers
- Manage images and volumes
- Monitor resource usage
- Deploy new containers

### cAdvisor (System Monitoring)

Start system monitoring with cAdvisor:

```bash
docker-compose up -d
```

Access monitoring dashboard at http://localhost:8172 to view:
- Container resource usage
- Host system metrics
- Docker daemon statistics

### Lazydocker (Terminal UI)

Use the terminal-based Docker manager:

```bash
lazydocker
```

This provides a curses-based interface for container management.

## Advanced Configuration

### Custom GPU Allocation

Control GPU resource allocation:

```bash
# Use specific GPUs
docker run --gpus device=0,1 --rm nvidia/cuda:11.0-base nvidia-smi

# Limit GPU memory
docker run --gpus device=0 --memory=4g --rm nvidia/cuda:11.0-base nvidia-smi
```

### Data Volume Mounting

Mount external datasets for training:

```bash
docker run --gpus all -it --rm \
  -v /path/to/your/data:/data \
  -v /path/to/models:/models \
  tensorflow/tensorflow:latest-gpu \
  python train.py
```

### Multi-GPU Training

Leverage multiple GPUs for distributed training:

```bash
docker run --gpus all -it --rm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  tensorflow/tensorflow:latest-gpu \
  python distributed_train.py
```

## Troubleshooting

### Common Issues and Solutions

**GPU Not Detected**:
- Verify drivers: `nvidia-smi`
- Check kernel modules: `lsmod | grep nvidia`
- Reinstall drivers if necessary

**Docker Permission Denied**:
```bash
sudo usermod -aG docker $USER
# Logout and login again, or reboot
```

**Container Fails to Start**:
- Check GPU passthrough: `docker run --gpus all --rm nvidia/cuda:11.0-base nvidia-smi`
- Verify image compatibility with your CUDA version

**Port Conflicts**:
- Modify port mappings: `-p 8889:8888` instead of `-p 8888:8888`
- Check running services: `netstat -tlnp | grep :8888`

**CUDA Compatibility Issues**:
- TensorFlow versions require specific CUDA/cuDNN versions
- Check compatibility matrix: https://www.tensorflow.org/install/source#gpu
- Use compatible image tags: `tensorflow/tensorflow:2.8.0-gpu`

**Memory Issues**:
- Increase Docker memory limits in Docker Desktop settings
- Use `--shm-size` for shared memory: `--shm-size=16g`

### Logs and Debugging

Access container logs for troubleshooting:

```bash
# View container logs
docker logs <container_name>

# Follow logs in real-time
docker logs -f <container_name>

# Check Docker daemon logs
sudo journalctl -u docker
```

### System Cleanup

If issues persist, clean and reinstall:

```bash
# Remove all containers and images
docker system prune -a

# Reinstall NVIDIA drivers
sudo apt purge nvidia-*
sudo ubuntu-drivers autoinstall
```

## Security Considerations

- **Container Isolation**: Run untrusted code in isolated containers
- **User Permissions**: Avoid running containers as root when possible
- **Network Security**: Use firewall rules to restrict access to management interfaces
- **Data Encryption**: Encrypt sensitive data before mounting volumes
- **Regular Updates**: Keep Docker images and host system updated

## Performance Optimization

### GPU Optimization
- Use latest NVIDIA drivers
- Enable persistence mode: `sudo nvidia-smi -pm 1`
- Monitor GPU utilization with `nvidia-smi -l 1`

### Docker Performance
- Use host networking for low-latency applications: `--net=host`
- Optimize volume mounts for I/O performance
- Use Docker build cache effectively

### System Tuning
- Disable swap for GPU workloads when possible
- Configure CPU affinity for container processes
- Monitor system resources with provided tools

## Contributing and Support

This project welcomes contributions. Please:
- Test changes on multiple Ubuntu versions
- Update documentation for any modifications
- Follow shell scripting best practices

For issues or questions:
- Check existing GitHub issues
- Provide system information and error logs
- Include steps to reproduce problems

## License

This project is provided as-is for educational and development purposes. Users are responsible for compliance with applicable licenses for all installed software.

---

Your Ubuntu system is now fully configured for high-performance AI development with Docker and NVIDIA GPU acceleration. Start building machine learning models with confidence in this optimized, containerized environment!