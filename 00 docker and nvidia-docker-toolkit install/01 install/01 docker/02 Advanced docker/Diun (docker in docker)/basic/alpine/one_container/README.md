# Docker-in-Docker Basic Setup

A simplified Docker-in-Docker (DinD) environment with SSH access, Portainer web interface, and a web-based terminal. This project allows you to run Docker containers inside a Docker container, providing a self-contained development or testing environment.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Services](#services)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Docker-in-Docker (DinD)**: Run Docker containers inside a Docker container
- **SSH Access**: Secure shell access to the container
- **Portainer Integration**: Web-based Docker management interface
- **Web Terminal**: Browser-based terminal using ttyd
- **Persistent Storage**: Data volumes for Docker and Portainer data
- **Customizable Configuration**: Environment-based setup

## Prerequisites

- Docker Engine installed on your host machine
- Docker Compose installed
- At least 2GB of available RAM (recommended for DinD)
- Basic knowledge of Docker and containerization

## Installation

1. **Clone or download this repository**:
   ```bash
   git clone <repository-url>
   cd docker_in_docker/basic
   ```

2. **Copy the environment configuration**:
   ```bash
   cp .env.example .env
   ```

3. **Build and start the services**:
   ```bash
   docker-compose up -d --build
   ```

   This command will:
   - Build the custom Docker image based on `docker:dind`
   - Start the container with all services
   - Create necessary data volumes

## Configuration

### Environment Variables

Edit the `.env` file to customize the setup:

- `SSH_PASSWORD`: Password for SSH root access (default: `password`)
  - **Security Note**: Change this to a strong password in production environments

### Container Configuration

- **Hostname**: The container is configured with hostname `wisrovi` for easy identification in networks

### Ports

The following ports are exposed on your host machine:

- `50421`: Portainer web interface
- `50422`: SSH access
- `50423`: Web terminal (ttyd)
- `50424`: HTTP (port 80 inside container)
- `50425`: HTTPS (port 443 inside container)

### Volumes

- `./data/dind-data`: Persistent storage for Docker data inside the container
- `./data/portainer_data`: Persistent storage for Portainer configuration and data

## Usage

### Starting the Environment

```bash
docker-compose up -d
```

### Accessing Services

1. **Portainer Web Interface**:
   - URL: `http://localhost:50421`
   - Default credentials: `admin` / `admin`
   - First login will prompt you to set a new password

2. **SSH Access**:
   ```bash
   ssh root@localhost -p 50422
   ```
   - Password: As set in `.env` (default: `password`)

3. **Web Terminal**:
   - URL: `http://localhost:50423`
   - Provides a browser-based shell interface

### Docker Operations Inside the Container

Once inside the container (via SSH or web terminal), you can run Docker commands:

```bash
# Check Docker version
docker --version

# Run a test container
docker run hello-world

# List running containers
docker ps

# Build and run your own containers
docker build -t my-app .
docker run -d my-app
```

### Managing the Environment

```bash
# View logs
docker-compose logs -f

# Access container shell
docker-compose exec dind-basic sh

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild the image
docker-compose build --no-cache
```

## Services

### Docker-in-Docker (DinD)

- Base image: `docker:dind`
- Runs Docker daemon inside the container
- Privileged mode required for DinD functionality

### SSH Server

- OpenSSH server running on port 50422
- Root login enabled with configurable password
- Provides secure remote access to the container

### Portainer

- Web-based Docker management interface
- Automatically installed and started inside the container
- Accessible at `http://localhost:50421`
- Manages both host and container Docker instances

### Web Terminal (ttyd)

- Browser-based terminal emulator
- Uses ttyd for web socket connections
- Accessible at `http://localhost:50423`
- Provides full shell access without SSH client

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   - Ensure ports 50421-50425 are not in use by other services
   - Modify port mappings in `docker-compose.yaml` if needed

2. **Permission Issues**:
   - DinD requires privileged mode, which may be restricted in some environments
   - Ensure Docker daemon on host allows privileged containers

3. **Memory Issues**:
   - DinD can be memory-intensive
   - Increase Docker memory limits if containers fail to start

4. **Portainer Not Accessible**:
   - Check if Portainer container is running: `docker-compose ps`
   - Verify port mapping and firewall settings

5. **SSH Connection Refused**:
   - Ensure SSH service is started inside the container
   - Check SSH password configuration
   - Verify port 50422 is accessible

### Logs and Debugging

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs dind-basic

# Access container for manual debugging
docker-compose exec dind-basic sh
```

### Resetting the Environment

If you encounter persistent issues:

```bash
# Stop and remove everything
docker-compose down -v --remove-orphans

# Remove built images
docker-compose rm -f

# Clean up unused Docker resources
docker system prune -f

# Restart fresh
docker-compose up -d --build
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.