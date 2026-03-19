# Docker on Docker

## Overview

This project provides a Docker container setup that enables running Docker commands from within a container. It utilizes the "Docker out of Docker" (DooD) approach by mounting the host's Docker socket into the container, allowing the container to interact with the host's Docker daemon without running a separate Docker daemon inside the container.

The setup is particularly useful for development environments, CI/CD pipelines, or scenarios where you need to build and manage Docker images from within a containerized environment.

## Features

- **Base Image**: Ubuntu 22.04 for stability and compatibility
- **SSH Access**: Built-in SSH server for remote container access
- **Docker CLI**: Pre-installed Docker command-line interface
- **Volume Mounting**: Persistent data storage and Docker socket access
- **Timezone Configuration**: Customizable timezone settings
- **Auto-restart**: Container automatically restarts on failure

## Prerequisites

Before using this project, ensure you have the following installed on your host machine:

- Docker (version 20.10 or later recommended)
- Docker Compose (version 1.29 or later)
- Basic knowledge of Docker concepts and command-line operations

## Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:

   ```bash
   cd /path/to/docker_on_docker
   ```

3. (Optional) Create a `files` directory to mount additional files into the container:

   ```bash
   mkdir files
   ```

   Place any files you want to access inside the container in this directory.

## Usage

### Starting the Container

To build and start the container, run:

```bash
docker-compose up -d
```

This command will:
- Build the Docker image from the provided Dockerfile
- Start the container in detached mode
- Map the container's SSH port to port 50422 on the host

### Accessing the Container

Connect to the container via SSH using:

```bash
ssh root@localhost -p 50422
```

**Default Credentials:**
- Username: `root`
- Password: `wZMqvW6aGt2omtedxz7s`

> **Security Warning**: The default root password is insecure. Change it immediately after first login using the `passwd` command.

### Running Docker Commands

Once inside the container, you can execute Docker commands as if you were on the host:

```bash
# List running containers
docker ps

# Run a test container
docker run hello-world

# Build an image
docker build -t my-image .

# Push to a registry
docker push my-registry/my-image
```

### Stopping the Container

To stop and remove the container:

```bash
docker-compose down
```

To stop without removing:

```bash
docker-compose stop
```

## Configuration

### Environment Variables

The container supports the following environment variables:

- `TZ`: Sets the timezone (default: `Europe/Madrid`)

To change the timezone, modify the `docker-compose.yaml` file:

```yaml
environment:
  - TZ=America/New_York
```

### Volumes

The setup mounts the following volumes:

- `/var/run/docker.sock:/var/run/docker.sock`: Grants access to the host's Docker daemon
- `./files:/app`: Mounts the local `files` directory for data persistence

### Ports

- **Host Port 50422** → **Container Port 22**: SSH access to the container

### Networking

The container runs with host networking disabled by default. All communication happens through the exposed ports.

## Architecture

### Dockerfile Breakdown

The `Dockerfile` performs the following operations:

1. **Base Setup**: Starts with Ubuntu 22.04 and installs essential tools
2. **SSH Configuration**: Installs and configures OpenSSH server with root access
3. **Docker CLI Installation**: Adds Docker's official repository and installs the Docker CLI
4. **Environment Setup**: Configures Python environment variables (for potential future use)
5. **Working Directory**: Sets `/app` as the working directory
6. **Startup Command**: Runs the SSH daemon as the main process

### Docker Compose Configuration

The `docker-compose.yaml` file defines:

- Service name: `docker_in_docker`
- Build context: Current directory
- Port mapping: 50422:22
- Volume mounts for Docker socket and data
- User: root
- Restart policy: always
- Timezone environment variable

## Security Considerations

1. **Docker Socket Access**: Mounting `/var/run/docker.sock` gives the container privileged access to the host's Docker daemon. This is powerful but risky.

2. **Root Access**: The container runs as root and allows SSH root login. Use caution in multi-user environments.

3. **Default Password**: Change the default root password immediately.

4. **Network Isolation**: Consider additional network restrictions for production use.

5. **Production Use**: This setup is designed for development. For production CI/CD, consider alternatives like Docker-in-Docker (DinD) with proper isolation.

## Troubleshooting

### Common Issues

1. **Permission Denied on Docker Commands**
   - Ensure the user running `docker-compose` has Docker permissions
   - Check that `/var/run/docker.sock` is accessible

2. **SSH Connection Refused**
   - Verify the container is running: `docker-compose ps`
   - Check SSH service status inside container: `systemctl status ssh` (if applicable)

3. **Container Won't Start**
   - Check build logs: `docker-compose build --no-cache`
   - View container logs: `docker-compose logs`

4. **Timezone Issues**
   - Verify `TZ` environment variable is set correctly
   - Restart container after changing timezone

### Debugging Commands

```bash
# View container logs
docker-compose logs

# Access container shell directly
docker-compose exec docker_in_docker bash

# Check container status
docker-compose ps

# Rebuild without cache
docker-compose build --no-cache
```

## Development and Customization

### Modifying the Dockerfile

To add additional tools or configurations:

1. Edit the `Dockerfile`
2. Rebuild the image: `docker-compose build`
3. Restart the container: `docker-compose up -d`

### Adding More Volumes

Edit `docker-compose.yaml` to add additional volume mounts:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - ./files:/app
  - ./my-data:/data
```

### Changing Ports

To use a different SSH port, modify the ports section in `docker-compose.yaml`:

```yaml
ports:
  - 2222:22
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is without any specific license. Please review the code and use at your own discretion.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review Docker and Docker Compose documentation
3. Open an issue in the repository (if applicable)

## Changelog

### Version 1.0
- Initial release
- Ubuntu 22.04 base
- SSH and Docker CLI setup
- Docker socket mounting
- Basic documentation