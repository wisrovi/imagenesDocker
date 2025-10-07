# DevOps Docker Configurations Repository

## Overview

This repository is a comprehensive collection of Docker-based configurations and setups for various DevOps tools and services. It provides ready-to-deploy environments for load balancing, code quality assurance, continuous integration/continuous deployment (CI/CD) pipelines, message queues, file sharing, monitoring, and more. Each subdirectory contains Docker Compose files, configuration files, and documentation to help you quickly spin up and manage DevOps infrastructure components.

The repository is designed to serve as a reference and starting point for building scalable, containerized DevOps environments. It covers essential aspects of modern software development workflows, from code quality checks to deployment pipelines and monitoring.

## Features

- **Load Balancing**: Nginx-based load balancer configurations with SSL support
- **Code Quality Assurance**: SonarQube, auto-healing services, uptime monitoring, and documentation tools
- **CI/CD Pipelines**: Jenkins and n8n workflow automation setups
- **Message Queues**: Celery, Kafka, Kurento, and MQTT implementations
- **File Sharing Services**: FTP, Nextcloud, Samba, and browser-based file access
- **Monitoring & Dashboards**: Heimdal dashboard and central logging
- **Docker-in-Docker**: Configurations for running Docker containers within containers
- **SSL Certificates**: Let's Encrypt and OpenSSL certificate management
- **Container Management**: Portainer for Docker container management

## Project Structure

```
.
├── balanceo_carga_nginx/          # Nginx load balancer with SSL
│   ├── certs/                     # SSL certificates
│   ├── config/                    # Nginx configuration files
│   ├── html/                      # Sample HTML pages
│   └── docker-compose.yaml        # Load balancer setup
├── central_logs/                  # Centralized logging service
├── Code QA/                       # Code quality and monitoring tools
│   ├── Check _services_status/    # Service health checks
│   │   ├── autohealth/            # Auto-healing services
│   │   ├── portained/             # Portainer container management
│   │   ├── uptime_kuma/           # Uptime monitoring
│   │   └── URL shortcuts/         # Dashboard shortcuts (Heimdal, Homer)
│   ├── Documentation/             # Documentation tools (MediaWiki, Snippet Box)
│   └── SonarQube/                 # Code quality analysis
├── Docker_over_docker/            # Docker-in-Docker configurations
├── Files shared/                  # File sharing services
│   ├── ftp/                       # FTP server
│   ├── netxcloud/                 # Nextcloud file sharing
│   ├── samba/                     # Samba file sharing
│   └── tcp in browser/            # File browser in browser
├── heimdall/                      # Heimdal dashboard
├── Pipelines/                     # CI/CD pipeline tools
│   ├── Jenkins/                   # Jenkins CI/CD server
│   └── n8n/                       # n8n workflow automation
├── Queues for services/           # Message queue systems
│   ├── celery/                    # Celery task queue
│   ├── kafka y zookeeper/         # Kafka message broker
│   ├── kurento/                   # Kurento media server
│   └── mqtt/                      # MQTT messaging
└── SSL_certificates/              # SSL certificate management
    ├── letsencript/               # Let's Encrypt certificates
    └── openssl/                   # OpenSSL certificates
```

## Prerequisites

Before using any of the configurations in this repository, ensure you have the following installed on your system:

- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 1.29 or later
- **Git**: For cloning repositories and version control
- **Bash/Shell**: For running scripts and commands

## Installation and Usage

Each subdirectory contains its own Docker Compose setup and documentation. Here's a general guide for getting started:

### General Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Navigate to Desired Service**:
   ```bash
   cd <service-directory>
   ```

3. **Review Configuration**:
   - Check the `docker-compose.yaml` file for service definitions
   - Review any configuration files in the `config/` directory
   - Read the README or documentation files for specific instructions

4. **Start the Service**:
   ```bash
   docker-compose up -d
   ```

5. **Access the Service**:
   - Refer to the service's documentation for access URLs and ports
   - Check `docker-compose.yaml` for port mappings

### Specific Service Guides

#### Load Balancing with Nginx
Navigate to `balanceo_carga_nginx/` and run:
```bash
docker-compose up -d
```
Access the load balancer at the configured ports (typically 80/443).

#### Jenkins CI/CD
Navigate to `Pipelines/Jenkins/` and run:
```bash
docker-compose up -d --build
```
Access Jenkins at `http://localhost:50443`.

#### SonarQube Code Analysis
Navigate to `Code QA/SonarQube/` and run:
```bash
docker-compose up -d
```
Access SonarQube at the configured port.

#### Kafka Message Broker
Navigate to `Queues for services/kafka y zookeeper/` and run:
```bash
docker-compose up -d
```
Use the provided Python examples for producers and consumers.

## Configuration

### Environment Variables
Many services use environment variables for configuration. Check each service's `docker-compose.yaml` file for required variables and modify them as needed.

### Volumes
Persistent data is stored using Docker volumes. Review volume mounts in `docker-compose.yaml` files to understand data persistence.

### Networking
Services are configured to work together through Docker networks. Some services may require custom network configurations for inter-service communication.

## Security Considerations

- **SSL/TLS**: Use the SSL certificate configurations for secure communications
- **Access Control**: Configure authentication and authorization for services
- **Network Security**: Implement proper firewall rules and network segmentation
- **Secret Management**: Avoid hardcoding sensitive information; use environment variables or Docker secrets
- **Updates**: Regularly update Docker images and dependencies for security patches

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If a port is already in use, modify the port mapping in `docker-compose.yaml`
2. **Permission Issues**: Ensure proper file permissions on mounted volumes
3. **Service Dependencies**: Some services depend on others; check logs for dependency errors
4. **Resource Limits**: Monitor resource usage and adjust Docker resource limits if needed

### Logs
View service logs with:
```bash
docker-compose logs -f <service-name>
```

### Stopping Services
To stop and remove containers:
```bash
docker-compose down
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-service`
3. Make your changes
4. Add documentation for new configurations
5. Test thoroughly
6. Submit a pull request with a clear description

### Contribution Guidelines

- Follow the existing directory structure
- Include comprehensive documentation (README files)
- Test configurations on multiple environments
- Use descriptive commit messages
- Update this main README when adding new services

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [SonarQube Documentation](https://docs.sonarsource.com/sonarqube/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Nginx Documentation](https://nginx.org/en/docs/)

## Support

For issues, questions, or contributions, please:

- Check existing documentation and README files
- Search for similar issues in the repository
- Create an issue with detailed information
- Provide logs and configuration when reporting problems

This repository aims to provide a solid foundation for DevOps infrastructure. Customize configurations to fit your specific needs and contribute back improvements for the community.