Services
========

This section details each service in the Docker stack.

Database (PostgreSQL)
---------------------

- **Image**: `postgres:15-alpine`
- **Purpose**: Stores user data, file metadata, and configurations.
- **Port**: Internal only (5432).
- **Volume**: `db_data`

.. note::
   PostgreSQL is chosen for its robustness and performance. For more info, visit `PostgreSQL Official Site <https://www.postgresql.org>`_.

Cache (Redis)
-------------

- **Image**: `redis:alpine`
- **Purpose**: Provides high-performance caching for sessions and data.
- **Port**: Internal only (6379).

.. tip::
   Redis improves response times significantly. Learn more at `Redis Official Site <https://redis.io>`_.

Nextcloud App
-------------

- **Image**: `nextcloud:latest`
- **Purpose**: Core application for file sharing and collaboration.
- **Port**: Internal (80).
- **Environment**: Configured via `.env`.

Features include:

- File synchronization
- User management
- App ecosystem

For full features, see `Nextcloud Features <https://nextcloud.com/features>`_.

OnlyOffice
----------

- **Image**: `onlyoffice/documentserver:latest`
- **Purpose**: Enables online document editing.
- **Port**: Internal (80).
- **Volume**: `onlyoffice_data`

.. warning::
   Requires JWT secret for security. Documentation at `OnlyOffice Docs <https://api.onlyoffice.com>`_.

Nginx Proxy
-----------

- **Image**: `nginx:alpine`
- **Purpose**: Reverse proxy with SSL termination.
- **Ports**: 80 (HTTP), 443 (HTTPS).
- **Volumes**: `nginx/conf.d`, `certs`

Handles:

- HTTP to HTTPS redirection
- SSL certificates
- Load balancing

Documentation Server
--------------------

- **Built from**: `./docs/Dockerfile`
- **Purpose**: Serves this Sphinx documentation.
- **Port**: 8080
- **Technology**: Nginx serving static HTML

.. note::
   Automatically rebuilt with documentation updates.

Service Dependencies
--------------------

The services have the following dependencies:

- Database (db) -> Nextcloud App
- Redis -> Nextcloud App
- Nextcloud App -> Nginx Proxy
- OnlyOffice -> Nginx Proxy
- Documentation Server (optional) -> Nginx Proxy

Nginx acts as the reverse proxy for app and onlyoffice, while docs serves documentation independently.

Monitoring Services
-------------------

To monitor services:

.. code-block:: bash

   docker-compose ps
   docker stats

For logs:

.. code-block:: bash

   docker-compose logs <service>

Replace `<service>` with db, redis, etc.