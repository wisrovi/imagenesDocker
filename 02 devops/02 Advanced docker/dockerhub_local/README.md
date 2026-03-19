# DockerHub Local

A comprehensive collection of local Docker registry solutions providing multiple deployment options for container image management in development and production environments.

## Overview

This project offers three distinct approaches to setting up local Docker registries, each catering to different needs and complexity levels:

- **Basic Registry**: A simple Docker Registry with web UI for basic image storage and management
- **Harbor**: An enterprise-grade registry with advanced security, vulnerability scanning, and management features
- **Portus**: A user-friendly frontend for Docker Registry with team-based access control

## Project Structure

```
dockerhub_local/
├── docker_registry/          # Basic Docker Registry setup with UI
│   ├── auth/                 # Authentication files
│   ├── config/               # Registry and nginx configuration
│   ├── scripts/              # Utility scripts for image management
│   ├── docker-compose.backend.yml
│   └── docker-compose.registry_ui.yml
├── harbor/                   # Harbor registry setup
│   ├── scripts/              # Image upload scripts
│   ├── docker-compose.yml    # Harbor services
│   ├── harbor.yml            # Harbor configuration
│   └── harbor-online-installer-v2.14.0.tgz
└── Portus/                   # Portus source code and development setup
    ├── app/                  # Rails application
    ├── bin/                  # Executables and scripts
    ├── config/               # Application configuration
    ├── db/                   # Database migrations
    ├── docker-compose.yml    # Development environment
    └── spec/                 # Test suite
```

## Components

### 1. Basic Docker Registry (`docker_registry/`)

A straightforward Docker Registry deployment with a web-based user interface for managing container images.

#### Features
- Docker Registry v2.8.2
- Web UI for image management (joxit/docker-registry-ui)
- Basic HTTP authentication
- Nginx reverse proxy with SSL termination
- Persistent storage for images
- Utility scripts for common operations

#### Quick Start

```bash
cd docker_registry
docker-compose -f docker-compose.backend.yml up -d
docker-compose -f docker-compose.registry_ui.yml up -d
```

#### Services
- **Registry**: Port 40231 (HTTP)
- **Web UI**: Port 40232 (HTTPS)

#### Authentication
- Registry: Username `wisrovi`, Password `nJ6OPitYMidApj8ebk4h`
- Web UI: Username `registry`, Password `wisrovi`

#### Available Scripts
- `push_image.sh <source> <destination>`: Push images to registry
- `pull_image.sh <image>`: Pull images from registry
- `list_images.sh`: List all images
- `test_registry.sh`: Test registry connectivity
- `test_frontend.sh`: Test web UI functionality
- `test_integration.sh`: Run integration tests
- `backup.sh`: Create registry backup

### 2. Harbor Registry (`harbor/`)

An enterprise-class container registry with advanced security and management capabilities.

#### Features
- Vulnerability scanning with Trivy
- Role-Based Access Control (RBAC)
- Image replication between registries
- Web-based management interface
- REST API for automation
- Audit logging
- Multi-architecture support
- Metrics and monitoring

#### Prerequisites
- Docker 20.10.10+
- Docker Compose 1.18.0+
- 4GB RAM minimum, 8GB recommended
- 40GB storage minimum

#### Installation

1. Extract Harbor installer:
```bash
tar -xzf harbor-online-installer-v2.14.0.tgz
cd harbor
```

2. Configure `harbor.yml`:
```yaml
hostname: your-registry.example.com
http:
  port: 80
harbor_admin_password: YourSecurePassword
```

3. Run installation:
```bash
./install.sh
```

4. Access Harbor at `http://your-registry.example.com`

#### Default Credentials
- Username: `admin`
- Password: `Harbor12345` (change immediately)

#### Image Management Scripts
- `1.sh` - `5.sh`: Individual image operations (pull, tag, push, clean, verify)
- `upload_multiple.sh`: Bulk upload popular images
- `upload_user1.sh` - `upload_user3.sh`: User-specific uploads

### 3. Portus (`Portus/`)

An open-source authorization server and user interface for Docker Registry, providing fine-grained access control.

#### Features
- Team-based namespaces and permissions
- LDAP authentication support
- OAuth/OpenID Connect integration
- Activity monitoring and audit logs
- Repository search and starring
- Application tokens for enhanced security
- User management (disable/enable accounts)

#### Development Setup

```bash
cd Portus
docker-compose up
```

#### Production Deployment

Portus supports multiple deployment configurations. Refer to the [official documentation](http://port.us.org/docs/deploy.html) for production setups.

#### Testing

```bash
# Unit tests
bundle exec rspec spec

# Frontend tests
yarn test

# Integration tests
./bin/test-integration.sh

# Code quality
bundle exec rubocop -a
yarn eslint
bundle exec brakeman
```

## Requirements

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Docker**: Version 20.10.10 or higher
- **Docker Compose**: Version 1.18.0 or higher (V2 compatible)
- **Hardware**:
  - CPU: 2 cores minimum
  - Memory: 4GB minimum (8GB+ recommended for Harbor)
  - Storage: 20GB+ for basic registry, 40GB+ for Harbor

### Network Requirements
- External hostname (not localhost/127.0.0.1) for Harbor
- Open ports for registry services (typically 80/443, 5000)

## Usage Examples

### Basic Registry Operations

```bash
# Login to registry
docker login localhost:40231

# Tag and push an image
docker tag nginx:latest localhost:40231/nginx:latest
docker push localhost:40231/nginx:latest

# Pull from registry
docker pull localhost:40231/nginx:latest
```

### Harbor Operations

```bash
# Login to Harbor
docker login your-registry.example.com

# Push image
docker tag myapp:latest your-registry.example.com/library/myapp:latest
docker push your-registry.example.com/library/myapp:latest

# Use Harbor API
curl -u admin:password https://your-registry.example.com/api/v2.0/projects
```

### Portus Operations

Portus provides a web interface for managing repositories, teams, and users. Access the web UI to:
- Create and manage namespaces
- Assign users to teams
- Set repository permissions
- Monitor registry activities

## Configuration

Each component has its own configuration files:

- **Basic Registry**: `docker_registry/config/registry-config.yml`, nginx.conf
- **Harbor**: `harbor/harbor.yml`
- **Portus**: `Portus/config/database.yml`, application settings

## Security Considerations

- Change default passwords immediately
- Enable HTTPS/TLS in production
- Configure firewalls appropriately
- Use strong authentication mechanisms
- Regularly scan images for vulnerabilities (Harbor)
- Implement proper access controls

## Backup and Recovery

### Basic Registry
```bash
cd docker_registry
./scripts/backup.sh
```

### Harbor
Harbor provides built-in backup tools. Refer to the Harbor documentation for detailed procedures.

### Portus
Backup the database and persistent volumes. Use standard Rails backup practices.

## Troubleshooting

### Common Issues

#### Registry Connection Problems
- Verify Docker daemon configuration
- Check network connectivity
- Ensure correct authentication credentials

#### Harbor Startup Issues
- Confirm hostname configuration
- Check resource availability (CPU, memory)
- Review Docker and Docker Compose versions

#### Portus Database Issues
- Verify database connectivity
- Check migration status
- Ensure proper environment variables

### Logs
- Basic Registry: Docker container logs
- Harbor: Centralized logging via rsyslog
- Portus: Rails application logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following each component's contribution guidelines
4. Test thoroughly
5. Submit a pull request

### Component-Specific Guidelines
- **Basic Registry**: Follow Docker best practices
- **Harbor**: Refer to [Harbor contributing guide](https://github.com/goharbor/harbor/blob/main/CONTRIBUTING.md)
- **Portus**: Follow [Portus contribution guidelines](https://github.com/SUSE/Portus/blob/master/CONTRIBUTING.md)

## License

This project contains multiple components with different licenses:

- **Basic Registry Setup**: MIT License (inferred from Docker Registry)
- **Harbor**: Apache License 2.0
- **Portus**: Apache License 2.0

Refer to individual component directories for specific license information.

## Support

### Documentation Links
- [Docker Registry](https://docs.docker.com/registry/)
- [Harbor Documentation](https://goharbor.io/docs/)
- [Portus Documentation](http://port.us.org/documentation.html)

### Community Support
- **Harbor**: [GitHub Issues](https://github.com/goharbor/harbor/issues), [Community](https://goharbor.io/community/)
- **Portus**: [Google Groups](https://groups.google.com/forum/#!forum/portus-dev)
- **Docker Registry**: [Docker Forums](https://forums.docker.com/)

## Version Information

- Docker Registry: 2.8.2
- Harbor: 2.14.0
- Portus: 2.4.3 (latest stable)

## Changelog

### v1.0.0
- Initial collection of local Docker registry solutions
- Basic Registry setup with web UI
- Harbor enterprise registry integration
- Portus frontend implementation

---

This project provides flexible options for local container registry deployment, from simple setups to enterprise-grade solutions. Choose the appropriate component based on your security, scalability, and management requirements.