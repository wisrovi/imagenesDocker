# Custom PostgreSQL Docker Image

## Architecture

See [architecture.md](architecture.md) for a detailed diagram of how all containers work together.

## Overview

This project provides a customized Docker setup for PostgreSQL 16.4. It builds a Docker image based on `postgres:16.4-bullseye` and includes several useful extensions and Python packages to create a powerful and extended database environment. The entire setup is orchestrated using Docker Compose for easy deployment and management.

The key features of this custom image are:
- **PostgreSQL 16.4**: Built on a stable and widely used version of PostgreSQL with custom performance tuning.
- **PostGIS**: Adds support for geographic objects to the PostgreSQL object-relational database.
- **PL/Python Support**: Includes the `plpython3u` extension, allowing you to write PostgreSQL functions and stored procedures in Python.
- **Python `requests` library**: Pre-installed to allow your Python functions within PostgreSQL to make HTTP requests to external services.
- **MySQL Foreign Data Wrapper**: Contains `mysql_fdw` to connect to and query remote MySQL databases from within PostgreSQL.
- **Performance Monitoring**: The `pg_stat_statements` extension is included to track planning and execution statistics of all SQL statements executed.
- **pg_cron**: Allows scheduling of PostgreSQL commands directly within the database.
- **pg_buffercache**: Provides information about the shared buffer cache.
- **Connection Pooling**: PgBouncer for efficient connection management.
- **Monitoring**: Prometheus and Grafana for real-time metrics and visualization.

- **Database Administration**: pgAdmin for graphical interface.
- **Automated Backups**: Cron-based daily backups.
- **Data Seeding**: Initial sample data loaded on first run.
- **Database Migrations**: Flyway for schema versioning and migrations.
- **RESTful API**: PostgREST for automatic API generation from database.
- **Caching**: Redis for high-performance data caching.
- **Documentation**: Sphinx documentation served in a container.
- **Logging**: ELK stack for centralized log management.
- **Health Checks**: Advanced health monitoring for all services.
- **Backups**: Incremental backups with automatic rotation.
- **Reverse Proxy**: Nginx proxy exposing all services through a single port (80).
- **Health Check**: Built-in health check using `pg_isready` for container monitoring.
- **Python Optimization**: Environment variables set to prevent `__pycache__` creation and ensure unbuffered output.
- **Security**: Credentials centralized in `.env` file; consider secrets for production.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## How to Use

1.  **Clone the repository or download the files.**
    
2.  **Navigate to the project directory:**
    ```sh
    cd /path/to/your/project/folder
    ```
    
3.  **Build and start the container:**
    Use Docker Compose to build the custom image and run the PostgreSQL container in detached mode:
    ```sh
    docker-compose up --build -d
    ```
    
4.  **Connect to the database:**
    You can connect to the PostgreSQL database using any standard client (like `psql`, DBeaver, or pgAdmin) with the following credentials:
     -   **Host**: `localhost`
     -   **Port**: `5433`
     -   **Database Name**: `test_eyesnroad`
    -   **User**: `perseus`
    -   **Password**: `7bPnWmNtzNF36c8J8yWN`
    
     Example connection using `psql`:
     ```sh
      psql -h localhost -p 5433 -U perseus -d test_eyesnroad
     ```

5.  **Stopping the container:**
    To stop the container, run:
    ```sh
    docker-compose down
    ```

## Backup and Restore

This setup includes scripts for backing up and restoring the database.

### Backup
To create a backup of the database, run:
```sh
./scripts/backup.sh
```
This will create a timestamped SQL file in the `backups/` directory.

### Restore
To restore from a backup file, run:
```sh
./scripts/restore.sh path/to/backup_file.sql
```
**Warning:** Restoring will overwrite the current database. Ensure you have a backup before restoring.

## Connection Pooling with PgBouncer

PgBouncer is included for connection pooling, improving scalability by reusing connections.

- **Port**: 6432
- Connect using the same credentials as PostgreSQL.

## Monitoring

The setup includes Prometheus for metrics collection and Grafana for visualization.

- **Prometheus**: Accessible at http://localhost:9090
- **Grafana**: Accessible at http://localhost:3000 (admin/admin)
- Add PostgreSQL dashboard in Grafana using the Prometheus data source.

## Database Administration

pgAdmin provides a web-based interface for managing PostgreSQL.

- **URL**: http://localhost:5050
- **Login**: admin@perseus.com / admin

## Automated Backups

Daily backups are scheduled at 2 AM using cron with compression and rotation.

- **Location**: `./backups/` directory
- Backups are created as `backup_YYYYMMDD_HHMMSS.sql` (compressed custom format)
- Automatic rotation keeps last 7 backups

## Security Scanning

Run security scans on the Docker image using Trivy to identify vulnerabilities.

```sh
./scripts/scan.sh
```

This script scans the built image for known security issues.

## Database Migrations

Use Flyway to manage database schema changes.

- Add SQL migration files to the `migrations/` directory following the naming convention `V{version}__{description}.sql`.
- Migrations run automatically on container startup.

## RESTful API

PostgREST provides an instant REST API for your PostgreSQL database.

- **URL**: http://localhost:3001
- Automatically generates endpoints for tables and views.
- Supports authentication and authorization.

## Caching

Redis is included for caching frequently accessed data.

- **Port**: 6379
- Use for session storage, query caching, or as a message broker.

## Documentation

Sphinx documentation is available in a dedicated container.

- **URL**: http://localhost:8000
- Built from reStructuredText files in the `docs/` directory.

## Logging

ELK stack for centralized logging.

- **Elasticsearch**: http://localhost:9200
- **Logstash**: Port 5044 (for log ingestion)
- **Kibana**: http://localhost:5601

Configure applications to send logs to Logstash.

## Reverse Proxy

Nginx reverse proxy provides access to all services through a single port. Individual service ports are commented out for security.

- **URL**: http://localhost
- Routes:
  - `/api` - PostgREST API
  - `/pgadmin` - pgAdmin
  - `/prometheus` - Prometheus
  - `/grafana` - Grafana
  - `/docs` - Documentation
  - `/elasticsearch` - Elasticsearch
  - `/kibana` - Kibana
  - `/pgbouncer` - PgBouncer

## Configuration

The configuration is managed through the `docker-compose.yaml` and `Dockerfile`.

### `docker-compose.yaml`

-   **Image Tag**: The built image is tagged as `wisrovi/postgres:v16.4`. You can change this in the `image` field.
-   **Credentials**: Credentials are managed using environment variables from the `.env` file. For production, consider using Docker secrets or external secret management.
-   **Data Persistence**: Named volumes are used for database and application data persistence, providing better portability and management compared to bind mounts.
-   **Resource Limits**: The service has resource reservations and limits defined to control its CPU and memory usage.
-   **Custom Config**: PostgreSQL uses a tuned `postgresql.conf` for optimal performance.
-   **Services**: Includes PgBouncer for pooling, Prometheus/Grafana for monitoring, and a read replica for HA.
-   **Networking**: The container is attached to a custom bridge network named `perseus-net`.

### `Dockerfile`

-   **Base Image**: The image is based on `postgres:16.4-bullseye`.
-   **Extensions**: SQL files in the `/docker-entrypoint-initdb.d` directory are automatically executed when the container starts for the first time. This is used to create the `plpython3u`, `pg_stat_statements`, `postgis`, `pg_cron`, and `pg_buffercache` extensions.
-   **System Packages**: `apt-get` is used to install additional packages like `python3`, `postgis`, `postgresql-16-mysql-fdw`, `postgresql-16-pg-cron`, and `postgresql-16-pg-buffercache`.
-   **Python Packages**: `pip3` is used to install Python libraries. Currently, `requests` is installed. You can add more packages by modifying the `Dockerfile`.

## Verifying Extensions

Once the container is running, you can connect to the `test_eyesnroad` database and verify that the extensions are enabled with the `\dx` command in `psql`.

```
test_eyesnroad=# \dx
                                          List of installed extensions
           Name          | Version |   Schema   |                        Description
------------------------+---------+------------+------------------------------------------------------------
  mysql_fdw              | 2.9.1   | public     | Foreign data wrapper for MySQL
  pg_buffercache         | 1.3     | public     | examine the shared buffer cache
  pg_cron                | 1.6     | pg_catalog | Job scheduler for PostgreSQL
  pg_stat_statements     | 1.8     | public     | track planning and execution statistics of all SQL statements executed
  plpgsql                | 1.0     | pg_catalog | PL/pgSQL procedural language
  plpython3u             | 1.0     | pg_catalog | PL/Python3U untrusted procedural language
  postgis                | 3.0.1   | public     | PostGIS geometry, geography, and raster spatial types and functions
(7 rows)
```

---
*This documentation was generated by Gemini.*
