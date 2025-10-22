# 🐳 Docker-in-Docker with Complete Monitoring

[![CI](https://github.com/your-repo/docker-dind-portainer/actions/workflows/ci.yml/badge.svg)](https://github.com/your-repo/docker-dind-portainer/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](http://localhost:8082)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

**Professional Docker-in-Docker (DinD) environment** with complete monitoring, integrated security, and total automation. This enterprise-ready platform includes **10 specialized containers** working together to provide a complete nested containerization experience.

### 🎯 What's Included

- **🏠 Docker-in-Docker**: DinD environment with internal Docker daemon
- **🎛️ Portainer**: Advanced web interface for container management
- **🔐 Secure SSH**: Remote access with automatic configuration
- **📊 Complete Monitoring**: Prometheus, Grafana, cAdvisor, Node Exporter
- **📝 Centralized Logging**: Loki + Promtail for unified logs
- **🚨 Alert System**: Alertmanager with intelligent notifications
- **🔧 REST API**: Programmatic container control
- **📚 Live Documentation**: Sphinx with integrated search
- **🔒 Advanced Security**: SSL, firewall, fail2ban, security headers
- **💾 Automatic Backups**: Backup system with retention

## 🏗️ Container Architecture

### Main Containers

| Container | Port | Function | Status |
|-----------|------|----------|--------|
| **dind** | 9003, 50422 | Docker-in-Docker with SSH and Portainer | 🟢 Primary |
| **docs-server** | 8082, 8443 | Sphinx documentation with SSL | 🟢 Documentation |
| **dind-api** | 5000 | REST API for container control | 🟢 API |

### Monitoring System

| Container | Port | Function | Status |
|-----------|------|----------|--------|
| **prometheus** | 9090 | Metrics collection | 🟢 Metrics |
| **grafana** | 3000 | Dashboards and visualization | 🟢 Dashboards |
| **node-exporter** | 9100 | Host system metrics | 🟢 System |
| **cadvisor** | 8080 | Container monitoring | 🟢 Containers |

### Logs and Alerts System

| Container | Port | Function | Status |
|-----------|------|----------|--------|
| **loki** | 3100 | Log storage | 🟢 Logs |
| **promtail** | - | Log collection | 🟢 Collector |
| **alertmanager** | 9093 | Alert management | 🟢 Alerts |

## 📋 System Requirements

### Minimum Requirements
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **RAM**: 8GB (16GB recommended)
- **CPU**: 2 cores (4+ recommended)
- **Disk**: 50GB free
- **OS**: Linux/macOS/Windows with WSL2

### Recommended Production Requirements
- **RAM**: 16GB+
- **CPU**: 4+ cores
- **Disk**: 100GB+ SSD
- **Network**: Stable connection for remote monitoring

## 🚀 Quick Start

### Automatic Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/your-repo/docker-dind-portainer.git
cd docker-dind-portainer

# Complete automatic setup
make setup
```

This automatically:
- ✅ Verifies prerequisites
- 🏗️ Builds optimized Docker images
- 📚 Generates complete documentation
- 🐳 Starts all 10 containers
- 🔍 Runs health checks
- 📊 Configures monitoring and alerts

### Manual Setup

```bash
# 1. Prepare configuration
cp .env.example .env
nano .env  # Configure secure credentials

# 2. Build and start services
docker-compose up -d

# 3. Verify status
docker-compose ps
```

## 🌐 Service Access

### Main Web Interfaces

| Service | URL | User | Password | Function |
|---------|-----|------|----------|----------|
| **Portainer** | http://localhost:9003 | admin | See .env | Docker Management |
| **Grafana** | http://localhost:3000 | admin | See .env | Dashboards |
| **Prometheus** | http://localhost:9090 | - | - | Metrics |
| **Documentation** | http://localhost:8082 | - | - | Guides |
| **Alertmanager** | http://localhost:9093 | - | - | Alerts |

### Command Line Access

```bash
# SSH to main container
ssh root@localhost -p 50422

# REST API for automation
curl http://localhost:5000/api/health

# Execute commands in containers
docker-compose exec dind docker ps
docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

4. **Install Portainer Inside the Container**

   Inside the container, install and run Portainer to manage the internal Docker environment:

   ```bash
   docker run -d -p 9000:9000 --name portainer \
       -v /var/run/docker.sock:/var/run/docker.sock \
       portainer/portainer-ce
   ```

   Access Portainer at `http://localhost:9003` (mapped from container's port 9000).

5. **Configure SSH Inside the Container**

   Inside the container (same session):

   ```bash
   # Install SSH and utilities
   apk add --no-cache openssh-server nano which tmux

   # Change the default root password
   echo "root:password" | chpasswd

   # Change the default SSH port
   sed -i 's/#Port 22/Port 50422/' /etc/ssh/sshd_config
   sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

   # Prepare SSH directories
   mkdir /run/sshd
   ssh-keygen -A
   chown root:root /var/empty
   chmod 755 /var/empty

   # Start SSH in a new tmux session
   tmux new -s ssh
   /usr/sbin/sshd -D -p 50422
   ```

   Detach from tmux with Ctrl+B, D.

## ⚙️ Advanced Configuration

### Environment Variables (.env)

```bash
# SSH Configuration
SSH_PASSWORD=Ch@ng3M3N0w!2024
SSH_PORT=50422

# Portainer Configuration
PORTAINER_ADMIN_PASSWORD=Adm1nP@ssw0rd!
PORTAINER_ADMIN_USERNAME=admin

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/ssl/certs
SSL_KEY_PATH=/etc/ssl/private
LETS_ENCRYPT_EMAIL=admin@example.com
DOMAIN_NAME=localhost

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=Gr@f@n@Adm1n!
METRICS_RETENTION_DAYS=30

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=7

# Security Configuration
FIREWALL_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
SESSION_TIMEOUT_MINUTES=30

# Notification Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
```

### Volume Architecture

```
volumes/
├── dind-data/          # Persistent Docker images and containers
├── portainer_data/     # Portainer configuration
├── grafana/           # Grafana dashboards and configuration
├── prometheus/        # Metrics and Prometheus configuration
├── loki/             # Centralized logs
├── ssl/              # SSL/TLS certificates
├── logs/             # Application logs
└── backups/          # Automatic backups
```

### Network Configuration

```yaml
# docker-compose.yaml includes custom network
networks:
  dind-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Resources and Limits

Each container has optimized limits:

- **DinD**: 2 CPUs, 4GB RAM
- **Monitoring**: 0.5 CPUs, 512MB-1GB RAM
- **Documentation**: 0.5 CPUs, 512MB RAM
- **API**: 0.5 CPUs, 256MB RAM

## 🎮 Usage and Operation

### Docker Container Management

```bash
# SSH access to DinD environment
ssh root@localhost -p 50422

# Inside container, normal Docker commands
docker ps                    # View containers
docker images               # View images
docker run hello-world      # Run test container
docker build -t my-app .    # Build image
```

### Monitoring and Observability

```bash
# View real-time metrics
open http://localhost:3000    # Grafana dashboards
open http://localhost:9090    # Prometheus metrics
open http://localhost:8080    # cAdvisor containers

# View centralized logs
open http://localhost:3100    # Loki log explorer

# Manage alerts
open http://localhost:9093    # Alertmanager
```

### Programmatic API

```bash
# Check health status
curl http://localhost:5000/api/health

# List containers
curl http://localhost:5000/api/containers

# Get container logs
curl http://localhost:5000/api/containers/myapp/logs

# Create backup
curl -X POST http://localhost:5000/api/backup
```

### Maintenance Operations

```bash
# Automatic backups
make backup

# View all service logs
docker-compose logs -f

# Run complete tests
make test

# Update documentation
make docs
```

## Examples

### Example 1: Building and Running a Custom Image

1. Create a Dockerfile in `./files/Dockerfile`:

   ```dockerfile
   FROM alpine:latest
   RUN apk add --no-cache nodejs npm
   WORKDIR /app
   COPY . .
   CMD ["node", "app.js"]
   ```

2. Inside the container:

   ```bash
   cd /app
   docker build -t my-app .
   docker run -p 8080:80 my-app
   ```

### Example 2: Using Docker Compose Inside DinD

Create a `docker-compose.yml` in `./files`:

```yaml
version: '3.8'
services:
  web:
    image: nginx:alpine
    ports:
      - "8080:80"
```

Then run:

```bash
cd /app
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   - Ensure the exposed ports (e.g., 50422 for SSH) are not in use by other services on your host.
   - Solution: Change the port mappings in `docker-compose.yaml`.

2. **Permission Denied**:
   - DinD requires privileged mode. If you encounter permission issues, ensure the container is running with `--privileged`.
   - On some systems, you may need to adjust Docker daemon settings.

3. **SSH Connection Refused**:
   - Verify SSH is running inside the container: `ps aux | grep sshd`
   - Check if the tmux session is active: `tmux ls`
   - Ensure the password is set correctly.

4. **Portainer Not Accessible**:
   - Confirm the container is running: `docker-compose ps`
   - Check logs: `docker-compose logs dind`
   - Ensure port 9003 is not blocked by firewall.

5. **Persistent Data Issues**:
   - If Docker data is not persisting, check the `./dind-data` volume permissions.
   - Solution: `sudo chown -R $USER:$USER ./dind-data`

### Logs and Debugging

- View container logs: `docker-compose logs -f dind`
- Access container shell: `docker exec -it dind sh`
- Check Docker daemon status inside container: `docker info`

## Security

⚠️ **Important**: Change default passwords in `.env` before production use.

- Default SSH password: `changeme123`
- Privileged container mode enabled
- TLS disabled for Docker daemon
- Root login allowed via SSH

See the [security documentation](http://localhost:8082/seguridad.html) for detailed recommendations.
- Use `userns_mode: "host"` for compatibility, but be aware of potential security risks.



## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Build documentation
make docs

# Serve docs locally
make docs-serve
```

### Project Structure

```
├── docker/              # Docker configuration
├── docs/                # Sphinx documentation
├── scripts/             # Setup and utility scripts
├── volumes/             # Persistent data
├── .env.example         # Environment template
├── docker-compose.yaml  # Service orchestration
└── Makefile            # Build automation
```

## Contributing

We welcome contributions! See our [contributing guide](http://localhost:8082/contribuyendo.html) for details.

### Quick Contribution

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues or questions, please open an issue in the repository or contact the maintainers.