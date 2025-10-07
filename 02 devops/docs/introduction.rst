Introduction
============

This repository is a comprehensive collection of Docker-based configurations and setups for various DevOps tools and services. It provides ready-to-deploy environments for load balancing, code quality assurance, continuous integration/continuous deployment (CI/CD) pipelines, message queues, file sharing, monitoring, and more. Each subdirectory contains Docker Compose files, configuration files, and documentation to help you quickly spin up and manage DevOps infrastructure components.

The repository is designed to serve as a reference and starting point for building scalable, containerized DevOps environments. It covers essential aspects of modern software development workflows, from code quality checks to deployment pipelines and monitoring.

Features
--------

- **Load Balancing**: Nginx-based load balancer configurations with SSL support
- **Code Quality Assurance**: SonarQube, auto-healing services, uptime monitoring, and documentation tools
- **CI/CD Pipelines**: Jenkins and n8n workflow automation setups
- **Message Queues**: Celery, Kafka, Kurento, and MQTT implementations
- **File Sharing Services**: FTP, Nextcloud, Samba, and browser-based file access
- **Monitoring & Dashboards**: Heimdal dashboard and central logging
- **Docker-in-Docker**: Configurations for running Docker containers within containers
- **SSL Certificates**: Let's Encrypt and OpenSSL certificate management
- **Container Management**: Portainer for Docker container management

Project Structure
-----------------

The repository is organized as follows:

.. code-block:: text

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

Prerequisites
-------------

Before using any of the configurations in this repository, ensure you have the following installed on your system:

- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 1.29 or later
- **Git**: For cloning repositories and version control
- **Bash/Shell**: For running scripts and commands

Installation and Usage
----------------------

Each subdirectory contains its own Docker Compose setup and documentation. Here's a general guide for getting started:

General Setup Steps
~~~~~~~~~~~~~~~~~~~

1. **Clone the Repository**:

   .. code-block:: bash

      git clone <repository-url>
      cd <repository-directory>

2. **Navigate to Desired Service**:

   .. code-block:: bash

      cd <service-directory>

3. **Review Configuration**:

   - Check the ``docker-compose.yaml`` file for service definitions
   - Review any configuration files in the ``config/`` directory
   - Read the README or documentation files for specific instructions

4. **Start the Service**:

   .. code-block:: bash

      docker-compose up -d

5. **Access the Service**:

   - Refer to the service's documentation for access URLs and ports
   - Check ``docker-compose.yaml`` for port mappings

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Many services use environment variables for configuration. Check each service's ``docker-compose.yaml`` file for required variables and modify them as needed.

Volumes
~~~~~~~

Persistent data is stored using Docker volumes. Review volume mounts in ``docker-compose.yaml`` files to understand data persistence.

Networking
~~~~~~~~~~

Services are configured to work together through Docker networks. Some services may require custom network configurations for inter-service communication.

Security Considerations
-----------------------

- **SSL/TLS**: Use the SSL certificate configurations for secure communications
- **Access Control**: Configure authentication and authorization for services
- **Network Security**: Implement proper firewall rules and network segmentation
- **Secret Management**: Avoid hardcoding sensitive information; use environment variables or Docker secrets
- **Updates**: Regularly update Docker images and dependencies for security patches