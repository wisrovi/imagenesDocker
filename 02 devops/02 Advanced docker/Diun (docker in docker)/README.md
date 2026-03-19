# 🐳 Docker-in-Docker Platform Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=flat&logo=terraform&logoColor=white)](https://terraform.io)

A comprehensive suite of Docker-in-Docker (DinD) environments designed for development, testing, and production containerization workflows. This project provides both basic and advanced implementations to cater to different use cases and complexity requirements.

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

This project offers a complete Docker-in-Docker ecosystem with two distinct implementations:

### **Basic Setup** (`basic/`)
Lightweight, straightforward DinD environments perfect for:
- Development and testing workflows
- CI/CD pipeline experimentation
- Learning Docker concepts
- Isolated container development

**Available Platforms:**
- **Alpine Linux**: Minimal footprint (~200MB), production-ready
- **Ubuntu**: Full compatibility (~500MB+), development-friendly

### **Advanced Setup** (`advanced/`)
Enterprise-grade DinD platform featuring:
- Comprehensive monitoring stack (Prometheus, Grafana, cAdvisor)
- Centralized logging (ELK Stack + Loki)
- REST API for programmatic management
- Security hardening and automation
- Infrastructure as Code with Terraform
- Live documentation with Sphinx

## 🏗️ Project Structure

```
docker-in-docker/
├── advanced/                 # Enterprise-grade DinD platform
│   ├── chaos/               # Chaos engineering experiments
│   ├── config/              # Configuration files
│   │   ├── grafana/         # Dashboards and data sources
│   │   ├── alertmanager.yml # Alert routing
│   │   ├── prometheus.yml   # Metrics collection
│   │   └── ...              # Additional configs
│   ├── docs/                # Sphinx documentation
│   ├── docker-compose.yaml  # 13-container orchestration
│   ├── .env.example         # Environment template
│   └── README.md            # Detailed advanced setup guide
├── basic/                   # Lightweight DinD implementations
│   ├── alpine/              # Alpine Linux version
│   │   ├── docs/            # Documentation
│   │   ├── docker-compose.yaml
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── ubuntu/              # Ubuntu version
│   │   ├── docs/            # Documentation
│   │   ├── docker-compose.yaml
│   │   ├── Dockerfile
│   │   └── README.md
│   └── README.md            # Basic setup overview
├── scripts/                 # Automation scripts
│   ├── backup.sh            # Automated backups
│   ├── setup_ssl.sh         # SSL certificate management
│   ├── install_portainer.sh # Portainer deployment
│   ├── test.sh              # Testing suite
│   └── ...                  # Additional utilities
├── terraform/               # Infrastructure as Code
│   ├── main.tf              # Terraform configuration
│   ├── variables.tf         # Variable definitions
│   └── outputs.tf           # Output specifications
├── test/                    # Testing infrastructure
│   ├── performance_test.jmx # JMeter performance tests
│   └── ...                  # Test suites
├── docs/                    # Project documentation
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation
│   └── ...                  # Additional docs
├── .env.example             # Global environment template
├── .gitignore               # Git ignore rules
├── docker-compose.yaml      # Root orchestration (if applicable)
├── Makefile                 # Build automation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Basic Setup (Recommended for beginners)

```bash
# Choose your preferred base OS
cd basic/alpine/    # Lightweight, production-ready
# OR
cd basic/ubuntu/    # Full compatibility, development-focused

# Configure environment
cp .env.example .env
# Edit .env with your preferences

# Launch the environment
docker-compose up -d --build

# Access services
# Portainer: http://localhost:50421
# SSH: ssh root@localhost -p 50422
# Web Terminal: http://localhost:50423
```

### Advanced Setup (Recommended for production)

```bash
cd advanced/

# Configure environment
cp .env.example .env
# Edit .env with secure credentials

# Complete automated setup
make setup

# Access enterprise services
# Portainer: http://localhost:9003
# Grafana: http://localhost:3000
# Documentation: http://localhost:8082
# API: http://localhost:5000
```

## ✨ Features

### Core Functionality
- **🐳 Docker-in-Docker**: Full containerization within containers
- **🔐 Secure Access**: SSH, SSL/TLS, and authentication
- **🌐 Web Interfaces**: Portainer for management, web terminals
- **💾 Persistent Storage**: Configurable volumes for data persistence
- **📚 Documentation**: Sphinx-powered guides and API docs

### Basic Implementation Features
- **Multi-Platform**: Alpine and Ubuntu variants
- **Simple Setup**: Minimal configuration required
- **Development Ready**: Perfect for learning and prototyping
- **Resource Efficient**: Lightweight container images

### Advanced Implementation Features
- **📊 Enterprise Monitoring**: Prometheus, Grafana, cAdvisor, Node Exporter
- **📝 Centralized Logging**: ELK Stack (Elasticsearch, Logstash, Kibana) + Loki
- **🚨 Intelligent Alerting**: Alertmanager with email notifications
- **🔧 REST API**: Complete programmatic container management
- **🔒 Security Hardening**: Firewall, fail2ban, rate limiting
- **💼 Infrastructure as Code**: Terraform deployment modules
- **🔄 CI/CD Ready**: GitHub Actions workflows included
- **🧪 Chaos Engineering**: Fault injection testing capabilities

## 📦 Installation

### Prerequisites

**System Requirements:**
- **Docker Engine**: 24.0+ (with BuildKit enabled)
- **Docker Compose**: 2.20+
- **RAM**: 4GB minimum (16GB+ recommended for advanced setup)
- **CPU**: 2 cores minimum (4+ recommended)
- **Disk**: 20GB free space (100GB+ for advanced with monitoring)
- **OS**: Linux, macOS, or Windows 10+ with WSL2

**Network Requirements:**
- Internet access for image downloads
- Ports 50421-50426 available (basic) or 3000-9600 (advanced)

### Basic Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd docker-in-docker
   ```

2. **Choose implementation:**
   ```bash
   # For lightweight production use
   cd basic/alpine/

   # For development with full tooling
   cd basic/ubuntu/
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

4. **Deploy:**
   ```bash
   docker-compose up -d --build
   ```

5. **Verify:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

### Advanced Installation

1. **Navigate to advanced directory:**
   ```bash
   cd advanced/
   ```

2. **Automated setup (recommended):**
   ```bash
   make setup
   ```

3. **Manual setup:**
   ```bash
   cp .env.example .env
   # Configure secure credentials
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```bash
   make health-check
   ```

### Terraform Deployment (Advanced)

For infrastructure as code deployment:

```bash
cd terraform/
terraform init
terraform plan
terraform apply -auto-approve
```

## 🎮 Usage

### Basic Usage

#### Accessing Services
```bash
# Portainer web interface
open http://localhost:50421

# SSH access
ssh root@localhost -p 50422

# Web-based terminal
open http://localhost:50423
```

#### Docker Operations
```bash
# Connect to container
docker-compose exec dind-basic sh  # Alpine
docker-compose exec dind-basic bash  # Ubuntu

# Inside container - run Docker commands
docker run hello-world
docker build -t my-app .
docker-compose up -d
```

### Advanced Usage

#### Service Access Points
```bash
# Core Services
Portainer:     http://localhost:9003
Grafana:       http://localhost:3000
Documentation: http://localhost:8082
REST API:      http://localhost:5000

# Monitoring Stack
Prometheus:    http://localhost:9090
cAdvisor:      http://localhost:8081
Node Exporter: http://localhost:9100

# Logging Stack
Loki:          http://localhost:3100
Kibana:        http://localhost:5601
Elasticsearch: http://localhost:9200

# Alerting
Alertmanager:  http://localhost:9093
```

#### Programmatic Management
```bash
# Health check
curl http://localhost:5000/api/health

# Container operations
curl http://localhost:5000/api/containers
curl -X POST http://localhost:5000/api/containers/{id}/start

# System information
curl http://localhost:5000/api/system/info
```

#### Automation Examples
```bash
# Automated backup
make backup

# Run test suite
make test

# Update documentation
make docs

# Security scan
make security-scan
```

## 🏛️ Architecture

### Basic Architecture

**Alpine Implementation:**
- Base: `docker:dind` (Alpine Linux)
- Size: ~200MB
- Init: dockerd-entrypoint.sh
- Best for: Production, resource constraints

**Ubuntu Implementation:**
- Base: `ubuntu:22.04` + Docker
- Size: ~500MB+
- Init: systemd-compatible
- Best for: Development, tool compatibility

**Shared Components:**
- Portainer CE for web management
- SSH server for remote access
- ttyd for web-based terminals
- Persistent volumes for data

### Advanced Architecture

**13 Specialized Containers:**

| Service | Purpose | Port |
|---------|---------|------|
| **dind** | Docker-in-Docker daemon | 9003, 9443, 50422 |
| **portainer** | Web management interface | 9000 |
| **prometheus** | Metrics collection | 9090 |
| **grafana** | Dashboards & visualization | 3000 |
| **cadvisor** | Container performance | 8081 |
| **node-exporter** | Host system metrics | 9100 |
| **loki** | Log aggregation | 3100 |
| **promtail** | Log collection | - |
| **elasticsearch** | Search & analytics | 9200 |
| **logstash** | Log processing | 5044 |
| **kibana** | Log visualization | 5601 |
| **alertmanager** | Alert routing | 9093 |
| **docs-server** | Live documentation | 8082 |

**Network Architecture:**
- Custom bridge networks for isolation
- Port forwarding for external access
- Volume mounting for persistence
- Privileged mode for DinD functionality

**Security Architecture:**
- fail2ban for intrusion prevention
- UFW firewall configuration
- SSL/TLS encryption
- Rate limiting and access controls

## ⚙️ Configuration

### Environment Variables

**Basic Setup (.env):**
```env
# SSH Configuration
SSH_PASSWORD=your_secure_password

# Portainer (change default credentials)
PORTAINER_ADMIN_PASSWORD=secure_password
```

**Advanced Setup (.env):**
```env
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

# Security Configuration
FIREWALL_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
SESSION_TIMEOUT_MINUTES=30
```

### Volume Management

**Basic Volumes:**
```
data/
├── dind-data/     # Docker daemon data
└── portainer_data/ # Portainer configuration
```

**Advanced Volumes:**
```
volumes/
├── dind-data/         # Docker images/containers
├── portainer_data/    # Portainer settings
├── grafana/          # Dashboards & data sources
├── prometheus/       # Metrics & configuration
├── loki/            # Centralized logs
├── elasticsearch/   # Search indexes
├── ssl/             # Certificates
├── logs/            # Application logs
└── backups/         # Automated archives
```

## 🔧 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :5042

# Modify docker-compose.yaml ports
ports:
  - "50425:9000"  # Change external port
```

#### Permission Issues
```bash
# Ensure privileged mode
docker-compose up --privileged dind

# Check Docker daemon status
docker-compose exec dind docker info
```

#### Memory Exhaustion
```bash
# Monitor resource usage
docker stats

# Adjust limits in docker-compose.yaml
deploy:
  resources:
    limits:
      memory: 2G
    reservations:
      memory: 1G
```

#### Service Access Problems
```bash
# Check service status
docker-compose ps

# View detailed logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart [service-name]
```

### Reset Procedures

**Basic Reset:**
```bash
docker-compose down -v
docker-compose up -d --build
```

**Advanced Reset:**
```bash
make clean
make setup
```

### Debug Commands

```bash
# Comprehensive logging
docker-compose logs

# Container inspection
docker-compose exec dind sh
docker info

# Network diagnostics
docker network ls
docker network inspect [network-name]

# Volume inspection
docker volume ls
docker volume inspect [volume-name]
```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information.

### Development Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/docker-in-docker.git
   cd docker-in-docker
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Setup Development Environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # For advanced development
   cd advanced/
   make setup-dev
   ```

4. **Code Quality**
   ```bash
   make lint
   make test
   ```

5. **Documentation**
   ```bash
   make docs
   ```

6. **Submit Pull Request**
   - Ensure all tests pass
   - Update documentation
   - Follow conventional commit format

### Areas for Contribution

- **New Features**: Additional monitoring integrations
- **Documentation**: Tutorials and troubleshooting guides
- **Testing**: Performance benchmarks and security tests
- **Infrastructure**: Terraform modules and CI/CD improvements
- **Security**: Hardening and compliance enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

- **Permitted**: Commercial use, modification, distribution, private use
- **Required**: License and copyright notice inclusion
- **Limitations**: No liability, no warranty

---

**Built with ❤️ for the containerization community**

*Empowering developers and organizations with comprehensive containerization solutions*

## 📞 Support & Resources

- **Documentation**: Check the `docs/` directory or live docs at http://localhost:8082 (advanced setup)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Community support and Q&A
- **Security**: Report security vulnerabilities privately

## 🙏 Acknowledgments

- **Docker Community**: For the amazing containerization platform
- **Portainer Team**: For excellent container management tools
- **Prometheus/Grafana**: For robust monitoring solutions
- **ELK Stack**: For powerful logging and analytics
- **Open Source Community**: For the tools and libraries that make this possible

---

*This project follows semantic versioning and is actively maintained.*