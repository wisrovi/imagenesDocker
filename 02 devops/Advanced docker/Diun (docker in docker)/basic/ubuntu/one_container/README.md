# Docker-in-Docker Basic Setup

A simplified Docker-in-Docker (DinD) environment built on Ubuntu, featuring SSH access, a web-based terminal, and Portainer for Docker management. This setup allows running Docker containers inside a Docker container, useful for development, testing, and CI/CD pipelines.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Services and Access](#services-and-access)
- [Architecture](#architecture)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Docker-in-Docker (DinD)**: Run Docker containers within a container using privileged mode.
- **SSH Access**: Secure shell access to the container for remote management.
- **Web Terminal**: Browser-based terminal interface using ttyd for easy access.
- **Portainer Integration**: Web UI for managing Docker containers, images, and networks.
- **Persistent Data**: Volumes for Docker data and Portainer configurations.
- **Customizable Ports**: Configurable port mappings for all services.

## Prerequisites

- Docker installed on your host machine.
- Docker Compose (version 3.8 or higher).
- At least 2GB of available RAM (recommended for DinD operations).
- Basic knowledge of Docker and containerization.

## Quick Start

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/docker-in-docker/basic/ubuntu
   ```

2. **Build and start the services**:
   ```bash
   docker-compose up -d
   ```

3. **Access the services**:
   - **Portainer**: Open http://localhost:50421 in your browser (default credentials: admin/admin)
   - **SSH**: `ssh root@localhost -p 50422` (password: password)
   - **Web Terminal**: Open http://localhost:50423 in your browser

4. **Verify the setup**:
   ```bash
   docker-compose logs -f dind-basic
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root to customize settings:

```env
SSH_PASSWORD=custom_password
```

### Docker Compose Configuration

The `docker-compose.yaml` file defines the service configuration:

- **Privileged Mode**: Required for DinD functionality.
- **Port Mappings**:
  - 50421:9000 - Portainer web interface
  - 50422:50422 - SSH server
  - 50423:7681 - ttyd web terminal
  - 50424:80 - HTTP (if needed)
  - 50425:443 - HTTPS (if needed)
  - 50426:9000 - Portainer agent (alternative port)
- **Volumes**:
  - `./data/dind-data:/var/lib/docker` - Persistent Docker data
  - `./data/portainer_data:/data` - Portainer data
- **Network**: Custom bridge network for isolation

### Dockerfile Details

The Docker image is based on Ubuntu 22.04 and includes:

- Docker Engine for DinD
- OpenSSH server for remote access
- ttyd for web-based terminal
- tmux and curl for additional utilities
- Non-interactive installation to avoid prompts

## Services and Access

### Portainer
- **URL**: http://localhost:50421
- **Purpose**: Web-based Docker management interface
- **Default Credentials**: admin / admin (change on first login)
- **Features**: Container management, image registry, network configuration

### SSH
- **Command**: `ssh root@localhost -p 50422`
- **Password**: password (or custom via SSH_PASSWORD env var)
- **Purpose**: Direct shell access to the container

### Web Terminal (ttyd)
- **URL**: http://localhost:50423
- **Purpose**: Browser-based terminal interface
- **Features**: Full terminal functionality in web browser

## Architecture

This setup creates a containerized environment where:

1. The main container runs Ubuntu with Docker installed
2. Docker daemon starts inside the container (DinD)
3. SSH and ttyd provide access methods
4. Portainer runs as a separate container inside the DinD environment
5. Data persistence through mounted volumes

The privileged mode allows the container to access host resources needed for Docker operations.

## Usage

### Basic Operations

```bash
# Start the services
docker-compose up -d

# View logs
docker-compose logs -f

# Access container shell
docker-compose exec dind-basic bash

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Using Docker Inside the Container

Once inside the container (via SSH or web terminal):

```bash
# Check Docker status
docker info

# Run a test container
docker run hello-world

# List containers
docker ps -a
```

### Managing Portainer

Access Portainer at http://localhost:50421 to:
- View and manage containers
- Pull and manage images
- Configure networks and volumes
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 50421-50426 are available on the host.
2. **Permission Issues**: Run with sudo if Docker requires elevated privileges.
3. **DinD Not Starting**: Check that the container has privileged mode enabled.
4. **SSH Connection Refused**: Verify SSH_PASSWORD is set correctly.

### Debugging Commands

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs dind-basic

# Restart services
docker-compose restart

# Clean up
docker-compose down -v
```

### Performance Considerations

- DinD can be resource-intensive; monitor host system resources.
- Use appropriate memory limits in production environments.
- Consider using Docker contexts for complex setups.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.