Overview
========

Introduction
------------

MinIO is a high-performance, distributed object storage system that is API-compatible with Amazon S3. It is designed to be lightweight, scalable, and easy to deploy, making it an excellent choice for modern cloud-native applications.

This repository provides two Docker Compose configurations for MinIO:

1. **MinIO-Normal**: A basic setup without SSL, suitable for development, testing, and local workflows.
2. **MinIO-SSL**: A secure setup with SSL/TLS encryption, including automated certificate generation, ideal for production or secure development environments.

Both configurations are optimized for integration with Data Version Control (DVC), a tool for versioning data and models in machine learning projects.

Key Benefits
------------

- **S3 Compatibility**: MinIO implements the S3 API, allowing you to use familiar tools and libraries designed for Amazon S3.
- **High Performance**: Features like erasure coding and bitrot protection ensure data integrity and efficient storage.
- **Scalability**: Easily scale from single-node deployments to distributed clusters.
- **Security**: The SSL variant provides encrypted communication and secure access.
- **Ease of Use**: Docker Compose makes deployment and management straightforward.

Architecture
------------

MinIO operates as a distributed object storage server. In these configurations:

- **MinIO Server**: The core storage server running in a Docker container.
- **Console**: A web-based UI for managing buckets, users, and policies.
- **SSL Layer** (SSL variant): Nginx or built-in SSL for encrypted connections.
- **Certificate Generation** (SSL variant): Automated OpenSSL-based certificate creation.

Use Cases
---------

- **Machine Learning Pipelines**: Store and version datasets and models with DVC.
- **Development Environments**: Quick local S3-compatible storage for testing applications.
- **Data Lakes**: Build scalable data lakes for analytics and processing.
- **Backup and Archiving**: Reliable, distributed storage for backups.
- **CI/CD Pipelines**: Store artifacts and build outputs.

Comparison of Setups
--------------------

.. list-table:: Setup Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - MinIO-Normal
     - MinIO-SSL
   * - SSL/TLS
     - No
     - Yes (self-signed certificates)
   * - Certificate Generation
     - N/A
     - Automated with OpenSSL
   * - Production Ready
     - Development/Testing
     - Production (with CA certs)
   * - Complexity
     - Low
     - Medium
   * - Ports
     - 30706 (API), 30707 (Console)
     - Same, with HTTPS

Prerequisites
-------------

- Docker (version 20.10 or later)
- Docker Compose (version 2.0 or later)
- Sufficient disk space for data storage
- Basic knowledge of Docker and containerization

Next Steps
----------

- :doc:`installation`: Detailed installation instructions
- :doc:`usage`: How to use the MinIO setups
- :doc:`examples`: Code examples and use cases