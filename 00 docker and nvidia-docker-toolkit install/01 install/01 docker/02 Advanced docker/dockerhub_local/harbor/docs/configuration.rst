Configuration
=============

harbor.yml
----------

The main configuration file contains settings for:

- **Network**: Hostname, ports, TLS certificates
- **Database**: PostgreSQL connection settings
- **Storage**: Backend storage (filesystem, S3, etc.)
- **Security**: Authentication, authorization
- **Scanning**: Trivy configuration
- **Logging**: Log levels and destinations
- **Cache**: Redis settings

Docker Compose
--------------

The ``docker-compose.yml`` defines all Harbor services. Key services include:

- ``nginx``: Reverse proxy
- ``core``: Main application
- ``registry``: Docker registry
- ``postgresql``: Database
- ``redis``: Cache
- ``jobservice``: Background jobs
- ``trivy``: Vulnerability scanner