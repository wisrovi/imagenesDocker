# Docker-in-Docker Basic Setup

A comprehensive Docker-in-Docker (DinD) environment providing isolated containerized development and testing environments. This project offers two implementations: one based on Alpine Linux for lightweight deployments and another on Ubuntu for broader compatibility and tooling.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Services](#services)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a self-contained Docker-in-Docker environment that allows you to run Docker containers inside Docker containers. It's particularly useful for:

- Development and testing environments
- CI/CD pipelines requiring Docker functionality
- Isolated container experimentation
- Educational purposes for learning Docker concepts

The setup includes SSH access, web-based terminal interface, and Portainer for visual Docker management.

## Features

- **Docker-in-Docker (DinD)**: Full Docker functionality within containers
- **Multi-Platform Support**: Alpine and Ubuntu-based implementations
- **SSH Access**: Secure remote shell access to containers
- **Web Terminal**: Browser-based terminal using ttyd
- **Portainer Integration**: Web UI for Docker management
- **Persistent Storage**: Configurable data volumes
- **Customizable Configuration**: Environment-based setup
- **Documentation**: Comprehensive Sphinx-based documentation
- **Build Automation**: Makefile support for common tasks

## Prerequisites

- **Docker Engine**: Version 20.10 or higher installed on host
- **Docker Compose**: Version 3.8 or higher
- **System Resources**: Minimum 2GB RAM (4GB recommended for DinD)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Network**: Access to Docker Hub for image pulls

## Project Structure

```
basic/
├── alpine/                 # Alpine Linux-based implementation
│   ├── docs/              # Sphinx documentation
│   ├── Dockerfile         # Alpine-based container definition
│   ├── docker-compose.yaml # Service orchestration
│   ├── start.sh          # Container initialization script
│   ├── Makefile          # Build automation
│   ├── .env.example      # Environment configuration template
│   └── README.md         # Alpine-specific documentation
├── ubuntu/                # Ubuntu-based implementation
│   ├── docs/             # Sphinx documentation
│   ├── Dockerfile        # Ubuntu-based container definition
│   ├── docker-compose.yaml # Service orchestration
│   ├── start.sh         # Container initialization script
│   ├── Makefile         # Build automation
│   └── README.md        # Ubuntu-specific documentation
└── README.md             # This file
```

## Installation

### Option 1: Alpine-based Setup (Recommended for Production)

```bash
# Navigate to Alpine directory
cd alpine/

# Copy environment configuration
cp .env.example .env

# Build and start services
docker-compose up -d --build
```

### Option 2: Ubuntu-based Setup (Recommended for Development)

```bash
# Navigate to Ubuntu directory
cd ubuntu/

# Build and start services
docker-compose up -d --build
```

### Verification

After installation, verify the setup:

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## Configuration

### Environment Variables

Create a `.env` file in your chosen implementation directory:

```env
# SSH Configuration
SSH_PASSWORD=your_secure_password

# Portainer Configuration
# Default credentials: admin/admin
# Change password on first login
```

### Port Mappings

Both implementations expose the following ports on the host:

- `50421`: Portainer web interface
- `50422`: SSH server
- `50423`: Web terminal (ttyd)
- `50424`: HTTP (port 80 inside container)
- `50425`: HTTPS (port 443 inside container)
- `50426`: Portainer agent (Ubuntu only)

### Data Persistence

The setup creates persistent volumes for:

- `./data/dind-data`: Docker daemon data and containers
- `./data/portainer_data`: Portainer configuration and settings

## Usage

### Starting the Environment

```bash
# Start services
docker-compose up -d

# Start with rebuild
docker-compose up -d --build
```

### Accessing Services

#### Portainer Web Interface
- **URL**: http://localhost:50421
- **Default Credentials**: admin / admin
- **Purpose**: Visual Docker management interface
- **Features**: Container monitoring, image management, network configuration

#### SSH Access
```bash
ssh root@localhost -p 50422
# Password: As configured in .env (default: password)
```

#### Web Terminal
- **URL**: http://localhost:50423
- **Purpose**: Browser-based shell interface
- **Features**: Full terminal functionality without SSH client

### Docker Operations Inside Container

Once inside the container:

```bash
# Verify Docker installation
docker --version
docker info

# Run test container
docker run hello-world

# List containers
docker ps -a

# Build custom images
docker build -t my-app .
docker run -d -p 8080:80 my-app
```

### Management Commands

```bash
# View logs
docker-compose logs -f

# Access container shell
docker-compose exec dind-basic sh  # Alpine
docker-compose exec dind-basic bash  # Ubuntu

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache
```

## Services

### Docker-in-Docker (DinD)
- **Base Images**: `docker:dind` (Alpine) / Ubuntu with Docker
- **Privileged Mode**: Required for container operations
- **Functionality**: Full Docker daemon inside container

### SSH Server
- **Software**: OpenSSH
- **Port**: 50422
- **Authentication**: Password-based (configurable)
- **Access**: Remote shell management

### Portainer
- **Version**: Community Edition (CE)
- **Interface**: Web-based management console
- **Features**: Container orchestration, image registry, monitoring
- **Integration**: Manages both host and container Docker instances

### Web Terminal (ttyd)
- **Technology**: WebSocket-based terminal emulator
- **Port**: 7681 (internal), 50423 (external)
- **Features**: Browser-based shell access, tmux support

## Architecture

### Alpine Implementation
- **Base Image**: `docker:dind` (Alpine Linux)
- **Size**: Lightweight (~200MB)
- **Package Manager**: apk
- **Init System**: Uses dockerd-entrypoint.sh
- **Best For**: Production deployments, resource-constrained environments

### Ubuntu Implementation
- **Base Image**: `ubuntu:22.04`
- **Size**: Larger (~500MB+)
- **Package Manager**: apt
- **Init System**: systemd-compatible
- **Best For**: Development, broader tool compatibility

### Network Architecture
- **Bridge Network**: Isolated container networking
- **Port Forwarding**: Host access to container services
- **Volume Mounting**: Persistent data storage
- **Privileged Access**: Required for DinD functionality

## Troubleshooting

### Common Issues

#### Port Conflicts
**Symptoms**: Services fail to start with port binding errors
**Solution**:
```bash
# Check port usage
netstat -tulpn | grep :5042

# Modify ports in docker-compose.yaml
# Change port mappings to available ports
```

#### Permission Denied
**Symptoms**: Docker operations fail inside container
**Cause**: Insufficient privileges or SELinux/AppArmor policies
**Solution**:
- Ensure privileged mode is enabled
- Check host Docker daemon permissions
- Disable SELinux if necessary (not recommended for production)

#### Memory Issues
**Symptoms**: Containers crash or become unresponsive
**Solution**:
- Increase Docker memory limits
- Monitor resource usage: `docker stats`
- Reduce concurrent container operations

#### Portainer Not Accessible
**Symptoms**: Web interface doesn't load
**Solution**:
```bash
# Check Portainer container status
docker-compose ps portainer

# View Portainer logs
docker-compose logs portainer

# Restart Portainer
docker-compose restart
```

#### SSH Connection Refused
**Symptoms**: SSH login fails
**Solution**:
- Verify SSH_PASSWORD in .env file
- Check SSH service status inside container
- Ensure port 50422 is accessible

### Debugging Commands

```bash
# Comprehensive logs
docker-compose logs

# Container resource usage
docker stats

# Access container for manual inspection
docker-compose exec dind-basic sh

# Docker daemon status inside container
docker-compose exec dind-basic docker info
```

### Reset Procedures

For persistent issues:

```bash
# Complete cleanup
docker-compose down -v --remove-orphans

# Remove images
docker-compose rm -f

# Clean system
docker system prune -f

# Fresh start
docker-compose up -d --build
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/your-feature`
3. **Make Changes**: Ensure code quality and testing
4. **Documentation**: Update relevant documentation
5. **Commit Changes**: `git commit -am 'Add feature description'`
6. **Push Branch**: `git push origin feature/your-feature`
7. **Pull Request**: Submit with detailed description

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd docker_in_docker/basic

# Choose implementation
cd alpine/  # or ubuntu/

# Install development dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
make docs-build

# Serve documentation locally
make docs-serve
```

## License

This project is licensed under the MIT License. See the LICENSE file in each implementation directory for details.

---

**Note**: This setup is designed for development and testing purposes. For production deployments, consider security hardening, resource limits, and monitoring solutions.