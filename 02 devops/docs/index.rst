DevOps Docker Configurations Documentation
==========================================

Welcome to the comprehensive documentation for the DevOps Docker Configurations repository. This project provides a collection of Docker-based setups for various DevOps tools and services, designed to help you quickly deploy and manage infrastructure components for modern software development workflows.

**Author:** `Wisrovi Rodriguez <https://es.linkedin.com/in/wisrovi-rodriguez>`_

**Version:** 1.0

**Release:** 1.0

Overview
--------

This repository includes configurations for:

- Load balancing with Nginx
- Code quality assurance tools (SonarQube, monitoring services)
- CI/CD pipelines (Jenkins, n8n)
- Message queues (Kafka, Celery, MQTT, Kurento)
- File sharing services (FTP, Nextcloud, Samba)
- Monitoring and dashboards (Heimdal, central logging)
- SSL certificate management
- Docker-in-Docker setups
- And more...

Getting Started
---------------

To get started with any of the configurations:

1. Ensure Docker and Docker Compose are installed
2. Navigate to the desired service directory
3. Run ``docker-compose up -d``
4. Follow the service-specific documentation

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   load_balancing
   code_quality
   pipelines
   message_queues
   file_sharing
   monitoring
   ssl_certificates
   docker_docker
   troubleshooting
   contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

