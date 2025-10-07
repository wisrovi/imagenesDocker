# Ubuntu AI Docker GPU Setup Guide

This guide transforms a fresh Ubuntu installation into a fully configured environment for AI development with TensorFlow, Docker, and GPU acceleration. Follow these steps sequentially for a complete setup.

## Prerequisites
- Fresh Ubuntu 18.04+ installation
- NVIDIA GPU (GTX/RTX series recommended)
- Internet connection
- Administrator privileges

## Quick Setup (Automated)

1. **Clone or download this repository** to your home directory:
   ```bash
   cd ~
   git clone <repository-url> ai-docker-setup
   cd ai-docker-setup
   ```

2. **Make scripts executable and run in order**:
   ```bash
   chmod +x scripts/*.sh
   ./scripts/install_docker.sh    # Installs Docker, NVIDIA drivers, and toolkit
   ./scripts/install_tools.sh     # Installs development tools and utilities
   ./scripts/install_containers.sh # Sets up containers and TensorFlow environment
   ```

3. **Reboot** when prompted to activate NVIDIA drivers.

4. **Verify installation**:
   ```bash
   nvidia-smi
   docker --version
   docker-compose --version
   ```

## Manual Setup (If Automated Fails)

If the automated scripts fail, run the individual scripts for specific components:

### Step 1: Docker and NVIDIA Setup
Run `./scripts/install_docker.sh` to install Docker, Docker Compose, NVIDIA drivers, and GPU toolkit.

### Step 2: Development Tools
Run `./scripts/install_tools.sh` to install code editors, browsers, Python packages, and system utilities.

### Step 3: Container Setup
Run `./scripts/install_containers.sh` to set up Portainer, test GPU integration, pull TensorFlow images, and launch Jupyter.

## Launch AI Development Environment

### Jupyter Notebook with GPU
```bash
docker run -u $(id -u):$(id -g) --gpus all -d --name tf-jupyter \
  -v ~/notebooks:/tf/notebooks \
  -p 8888:8888 -p 6006:6006 \
  --user root -e GRANT_SUDO=yes \
  tensorflow/tensorflow:latest-gpu-jupyter
```

Access at: http://localhost:8888

### Interactive TensorFlow Shell
```bash
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu bash
```

## Additional Tools

### Portainer (Docker Management UI)
```bash
docker run -d -p 9000:9000 --name portainer \
  --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce
```
Access at: http://localhost:9000

### System Monitoring
```bash
docker-compose up -d  # Starts cAdvisor on port 8172
```
Access at: http://localhost:8172

### Development Tools
```bash
sudo apt install -y stacer bleachbit timeshift  # System monitoring, cleaner, backup
```

## Troubleshooting

- **GPU not detected**: Run `nvidia-smi` and ensure drivers are installed
- **Docker permission denied**: `sudo usermod -aG docker $USER` then logout/login
- **Port conflicts**: Change port mappings in docker run commands
- **CUDA errors**: Check TensorFlow-CUDA compatibility matrix

## Project Structure

- `scripts/`: Contains installation scripts for Docker, tools, and containers
  - `install_docker.sh`: Installs Docker, Docker Compose, NVIDIA drivers, and toolkit
  - `install_tools.sh`: Installs development tools like VS Code, Chrome, Python packages, etc.
  - `install_containers.sh`: Sets up Portainer, tests GPU, pulls TensorFlow images, and launches Jupyter
  - `README.md`: Documentation for the scripts
- `docker-compose.yml`: Configuration for cAdvisor system monitoring container
- `README_SETUP.md`: This setup guide

## What's Included

This setup provides:
- GPU-accelerated TensorFlow environment
- Jupyter notebooks for interactive development
- TensorBoard for model visualization
- Docker containerization for reproducible environments
- System monitoring with cAdvisor
- Portainer for easy container management
- Development tools and utilities

## Next Steps

1. Start developing AI models in Jupyter at http://localhost:8888
2. Monitor your system with cAdvisor at http://localhost:8172
3. Manage containers with Portainer at http://localhost:9000
4. Mount your datasets: `-v /path/to/data:/data`
5. Scale with multiple GPUs: `--gpus all`
6. Backup your work regularly with Timeshift

Your Ubuntu system is now optimized for AI development with Docker and GPU support!