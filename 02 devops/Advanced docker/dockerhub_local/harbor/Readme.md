# Harbor Container Registry Setup

## Overview

This project provides a complete setup for deploying and managing a Harbor container registry instance. Harbor is an open-source container image registry that secures images with role-based access control, scans images for vulnerabilities, and signs images as trusted. This setup includes Docker Compose configurations, installation scripts, and utility scripts for image management.

Harbor extends the Docker Distribution by adding functionalities usually required by enterprises such as security, identity, and management. It supports features like vulnerability scanning with Trivy, image replication, and user management.

## Features

- **Container Registry**: Store and distribute container images
- **Security Scanning**: Integrated Trivy vulnerability scanner
- **Role-Based Access Control (RBAC)**: Manage user permissions
- **Image Replication**: Sync images between registries
- **Web UI**: User-friendly interface for management
- **API Support**: RESTful API for automation
- **Multi-architecture Support**: Handle different CPU architectures
- **Audit Logging**: Track user actions and system events

## Architecture

The Harbor setup consists of several components running as Docker containers:

- **Proxy (Nginx)**: Handles incoming requests and load balancing
- **Core**: Main API server for Harbor
- **Registry**: Docker Distribution registry backend
- **Registry Controller**: Manages registry operations
- **Database (PostgreSQL)**: Stores metadata
- **Redis**: Caching and session storage
- **Job Service**: Handles background tasks
- **Log**: Centralized logging
- **Portal**: Web UI
- **Exporter**: Metrics collection
- **Trivy Adapter**: Vulnerability scanning (optional)

## Prerequisites

Before installing Harbor, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows
- **Docker**: Version 20.10.10 or higher
- **Docker Compose**: Version 1.18.0 or higher (or Docker Compose V2)
- **Hardware Requirements**:
  - CPU: 2 cores minimum, 4 cores recommended
  - Memory: 4GB minimum, 8GB recommended
  - Storage: 40GB minimum for data persistence
- **Network**: Accessible hostname (not localhost/127.0.0.1)

## Installation

### Step 1: Download Harbor

Download the Harbor installer package from the official releases:

```bash
# Download from GitHub releases
wget https://github.com/goharbor/harbor/releases/download/v2.14.0/harbor-online-installer-v2.14.0.tgz
tar -xzf harbor-online-installer-v2.14.0.tgz
cd harbor
```

### Step 2: Configure Harbor

Edit the `harbor.yml` configuration file:

```yaml
hostname: your-domain.com
http:
  port: 80
harbor_admin_password: YourSecurePassword
database:
  password: YourDBPassword
```

Key configuration options:
- `hostname`: External hostname for Harbor (required)
- `http.port`: HTTP port (default: 80)
- `harbor_admin_password`: Initial admin password
- `data_volume`: Directory for persistent data (default: ./data)

### Step 3: Run Installation

Execute the installation script:

```bash
./install.sh
```

This script will:
1. Check Docker and Docker Compose installation
2. Load Harbor images (if offline package used)
3. Prepare configuration files
4. Start all Harbor services using Docker Compose

### Step 4: Access Harbor

Once installation completes:
- Web UI: http://your-domain.com
- Default admin credentials: admin / Harbor12345 (change immediately)

## Configuration

### harbor.yml

The main configuration file contains settings for:

- **Network**: Hostname, ports, TLS certificates
- **Database**: PostgreSQL connection settings
- **Storage**: Backend storage (filesystem, S3, etc.)
- **Security**: Authentication, authorization
- **Scanning**: Trivy configuration
- **Logging**: Log levels and destinations
- **Cache**: Redis settings

### Docker Compose

The `docker-compose.yml` defines all Harbor services. Key services include:

- `nginx`: Reverse proxy
- `core`: Main application
- `registry`: Docker registry
- `postgresql`: Database
- `redis`: Cache
- `jobservice`: Background jobs
- `trivy`: Vulnerability scanner

## Usage

### Basic Operations

#### Push an Image

```bash
# Tag image for Harbor
docker tag my-image:latest your-registry.com/library/my-image:latest

# Login to Harbor
docker login your-registry.com

# Push image
docker push your-registry.com/library/my-image:latest
```

#### Pull an Image

```bash
docker pull your-registry.com/library/my-image:latest
```

### Using the Scripts

This setup includes utility scripts in the `scripts/` directory for automated image management:

#### Individual Scripts

- `1.sh`: Pull image from Docker Hub
- `2.sh`: Tag image for Harbor registry
- `3.sh`: Push image to Harbor
- `4.sh`: Remove local image copy
- `5.sh`: Pull image from Harbor registry

Usage example:

```bash
./scripts/1.sh nginx latest
./scripts/2.sh nginx latest
./scripts/3.sh nginx latest
./scripts/4.sh nginx latest
./scripts/5.sh nginx latest
```

#### Bulk Upload Script

`upload_multiple.sh` automates the process for multiple images:

```bash
./scripts/upload_multiple.sh
```

This script processes a predefined list of popular images (Python, Node.js, Java, etc.) and uploads them to Harbor.

### API Usage

Harbor provides a REST API for programmatic access. Examples:

#### List Projects

```bash
curl -u admin:password http://your-registry.com/api/v2.0/projects
```

#### List Repositories

```bash
curl -u admin:password http://your-registry.com/api/v2.0/projects/library/repositories
```

#### List Artifacts

```bash
curl -u admin:password http://your-registry.com/api/v2.0/projects/library/repositories/nginx/artifacts
```

## Vulnerability Scanning

Harbor integrates Trivy for automatic vulnerability scanning:

1. Enable scanning in `harbor.yml`
2. Push an image to Harbor
3. View scan results in the Web UI
4. Set policies for blocking vulnerable images

## Replication

Configure replication to sync images between registries:

1. Go to Administration > Registries
2. Add source/destination registries
3. Create replication rules
4. Schedule automatic replication

## Monitoring

### Metrics

Harbor exposes Prometheus metrics on port 9090:

```bash
curl http://your-registry.com/metrics
```

### Logs

Logs are centralized using rsyslog. View logs:

```bash
docker logs harbor-log
```

## Troubleshooting

### Common Issues

#### Harbor Won't Start

- Check Docker and Docker Compose versions
- Verify hostname configuration
- Ensure ports are not in use
- Check disk space

#### Cannot Push Images

- Verify user permissions
- Check project settings
- Ensure correct image tagging
- Validate network connectivity

#### Database Connection Issues

- Check PostgreSQL container status
- Verify database credentials in `harbor.yml`
- Ensure data volume permissions

### Logs and Debugging

Enable debug logging in `harbor.yml`:

```yaml
log:
  level: debug
```

View service logs:

```bash
docker-compose logs [service-name]
```

### Reset Harbor

To completely reset Harbor:

```bash
docker-compose down -v
rm -rf data/
./install.sh
```

## Security Best Practices

- Change default admin password
- Enable HTTPS/TLS
- Configure firewall rules
- Use strong passwords
- Regularly update Harbor
- Scan images for vulnerabilities
- Implement RBAC policies
- Monitor audit logs

## Backup and Restore

### Backup

```bash
# Stop Harbor
docker-compose down

# Backup data directory
tar -czf harbor-backup.tar.gz data/

# Backup database
docker run --rm -v harbor_data:/data -v $(pwd):/backup \
  postgres:13 pg_dump -h harbor-db -U postgres registry > backup.sql
```

### Restore

```bash
# Restore data
tar -xzf harbor-backup.tar.gz

# Restore database
docker run --rm -v harbor_data:/data -v $(pwd):/backup \
  postgres:13 psql -h harbor-db -U postgres registry < backup.sql

# Start Harbor
docker-compose up -d
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Support

- **Documentation**: https://goharbor.io/docs/
- **GitHub Issues**: https://github.com/goharbor/harbor/issues
- **Community**: https://goharbor.io/community/

## Releases

For the latest releases and changelogs, visit:
https://github.com/goharbor/harbor/releases