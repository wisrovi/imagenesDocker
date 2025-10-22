Architecture
============

Container Diagram
-----------------

.. code-block::

   +-------------------+     +-------------------+     +-------------------+
   |   PostgreSQL      |     |   PgBouncer       |     |   PostgREST       |
   |   (Port 5433)     |<--->|   (Port 6432)     |     |   (Port 3001)     |
   |   Database        |     |   Connection Pool |     |   REST API        |
   +-------------------+     +-------------------+     +-------------------+
             |                         |                         |
             |                         |                         |
             v                         v                         v
   +-------------------+     +-------------------+     +-------------------+
   |   pgAdmin         |     |   Prometheus      |     |   Grafana         |
   |   (Port 5050)     |     |   (Port 9090)     |     |   (Port 3000)     |
   |   Admin GUI       |     |   Metrics         |     |   Dashboards      |
   +-------------------+     +-------------------+     +-------------------+
             |                         |
             |                         |
             v                         v
   +-------------------+     +-------------------+
   |   Flyway          |     |   Backup Service  |
   |   Migrations      |     |   Auto Backups    |
   +-------------------+     +-------------------+
             |
             |
             v
   +-------------------+
   |   Redis           |
   |   (Port 6379)     |
   |   Cache           |
   +-------------------+

Container Descriptions
----------------------

- **PostgreSQL**: Main database server with custom extensions and tuned configuration.
- **PgBouncer**: Connection pooler for PostgreSQL to improve performance and scalability.
- **PostgREST**: Automatically generates a REST API from PostgreSQL database schema.
- **pgAdmin**: Web-based administration interface for PostgreSQL.
- **Prometheus**: Monitoring system that collects metrics from PostgreSQL exporter.
- **Grafana**: Visualization tool for creating dashboards from Prometheus metrics.
- **Flyway**: Database migration tool for versioning and applying schema changes.
- **Backup Service**: Automated daily backup service using cron and pg_dump.
- **Redis**: In-memory data structure store used for caching.