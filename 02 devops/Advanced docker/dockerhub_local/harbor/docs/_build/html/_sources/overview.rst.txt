Overview
========

This project provides a complete setup for deploying and managing a Harbor container registry instance. Harbor is an open-source container image registry that secures images with role-based access control, scans images for vulnerabilities, and signs images as trusted. This setup includes Docker Compose configurations, installation scripts, and utility scripts for image management.

Harbor extends the Docker Distribution by adding functionalities usually required by enterprises such as security, identity, and management. It supports features like vulnerability scanning with Trivy, image replication, and user management.

Features
--------

- **Container Registry**: Store and distribute container images
- **Security Scanning**: Integrated Trivy vulnerability scanner
- **Role-Based Access Control (RBAC)**: Manage user permissions
- **Image Replication**: Sync images between registries
- **Web UI**: User-friendly interface for management
- **API Support**: RESTful API for automation
- **Multi-architecture Support**: Handle different CPU architectures
- **Audit Logging**: Track user actions and system events

Architecture
------------

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

Prerequisites
-------------

Before installing Harbor, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows
- **Docker**: Version 20.10.10 or higher
- **Docker Compose**: Version 1.18.0 or higher (or Docker Compose V2)
- **Hardware Requirements**:

  - CPU: 2 cores minimum, 4 cores recommended
  - Memory: 4GB minimum, 8GB recommended
  - Storage: 40GB minimum for data persistence
- **Network**: Accessible hostname (not localhost/127.0.0.1)