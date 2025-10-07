# Installation Scripts

This directory contains a set of bash scripts to automate the installation and configuration of a complete development environment for a machine learning project using Docker and NVIDIA GPUs.

## Scripts

-   `install_docker.sh`: This script automates the installation of Docker, Docker Compose, and the NVIDIA Docker Toolkit. It also installs the appropriate NVIDIA drivers for your system.

-   `install_tools.sh`: This script installs a variety of development tools and system utilities, including code editors, web browsers, and system monitoring tools.

-   `install_containers.sh`: This script sets up the Docker containers required for the project, including Portainer for container management, and TensorFlow with GPU support for machine learning development.

## Usage

The scripts should be executed in the following order:

1.  `./install_docker.sh`
2.  `./install_tools.sh`
3.  `./install_containers.sh`

**Note:** These scripts require `sudo` privileges to install packages and modify system settings. Make sure to make them executable before running:

```bash
chmod +x *.sh
```
