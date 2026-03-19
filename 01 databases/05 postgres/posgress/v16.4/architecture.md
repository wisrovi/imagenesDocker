# Container Architecture

## Functionality Diagram

```
+-------------------+     +-------------------+     +-------------------+
|   PostgreSQL      |     |   PgBouncer       |     |   PostgREST       |
|   (/postgres)     |<--->|   (/pgbouncer)    |     |   (/api)          |
|   Database        |     |   Connection Pool |     |   REST API        |
+-------------------+     +-------------------+     +-------------------+
          |                         |                         |
          |                         |                         |
          v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   pgAdmin         |     |   Prometheus      |     |   Grafana         |
|   (/pgadmin)      |     |   (/prometheus)   |     |   (/grafana)      |
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
|   (/redis)        |
|   Cache           |
+-------------------+
          |
          |
          v
+-------------------+
|   Docs            |
|   (/docs)         |
|   Sphinx Docs     |
+-------------------+
          |
          |
          v
+-------------------+     +-------------------+     +-------------------+
|   Elasticsearch   |     |   Logstash        |     |   Kibana          |
|   (/elasticsearch)|     |   (/logstash)     |     |   (/kibana)       |
|   Search Engine   |     |   Log Processor   |     |   Visualization   |
+-------------------+     +-------------------+     +-------------------+
          |
          |
          v
+-------------------+
|   Nginx           |
|   (Port 80)       |
|   Reverse Proxy   |
+-------------------+
```

## Container Descriptions

- **PostgreSQL**: Main database server with custom extensions and tuned configuration.
- **PgBouncer**: Connection pooler for PostgreSQL to improve performance and scalability.
- **PostgREST**: Automatically generates a REST API from PostgreSQL database schema.
- **pgAdmin**: Web-based administration interface for PostgreSQL.
- **Prometheus**: Monitoring system that collects metrics from PostgreSQL exporter.
- **Grafana**: Visualization tool for creating dashboards from Prometheus metrics.
- **Flyway**: Database migration tool for versioning and applying schema changes.
- **Backup Service**: Automated daily backup service using cron and pg_dump with rotation.
- **Redis**: In-memory data structure store used for caching.
- **Docs**: Container serving Sphinx-generated documentation.
- **Elasticsearch**: Distributed search and analytics engine for logs.
- **Logstash**: Server-side data processing pipeline for log ingestion.
- **Kibana**: Web interface for visualizing Elasticsearch data.
- **Nginx**: Reverse proxy server routing requests to appropriate services.

## Flow Description

1. **PostgreSQL**: Main database with extensions and custom configuration.
2. **PgBouncer**: Manages connection pooling for better performance.
3. **PostgREST**: Generates REST API automatically from PostgreSQL tables.
4. **pgAdmin**: Graphical interface for database administration.
5. **Prometheus**: Collects metrics from PostgreSQL via exporter.
6. **Grafana**: Visualizes metrics in customizable dashboards.
7. **Flyway**: Runs database schema migrations.
8. **Backup Service**: Performs automated daily backups.
9. **Redis**: Provides caching for frequently accessed data.
10. **Docs**: Serves the project documentation built with Sphinx.
11. **Elasticsearch**: Stores and indexes log data for search and analysis.
12. **Logstash**: Collects, processes, and forwards logs to Elasticsearch.
13. **Kibana**: Provides visualization and exploration of log data.
14. **Nginx**: Acts as a gateway, routing external requests to internal services.

## Connections

- Applications connect to PgBouncer (6432) or directly to PostgreSQL (5433).
- REST API accessible via PostgREST (3001).
- Monitoring via Prometheus (9090) and Grafana (3000).
- Administration via pgAdmin (5050).
- Caching via Redis (6379).
- Logging via ELK stack (9200, 5044, 5601).
- All services accessible via Nginx proxy (80).

## Volumes

- `postgres_data`: PostgreSQL data.
- `grafana_data`: Grafana configurations.
- `pgadmin_data`: pgAdmin data.
- `redis_data`: Redis data.
- `elasticsearch_data`: Elasticsearch data.
- Bind mounts: `./backups`, `./config`, `./migrations`, `./scripts`.

## Access Paths

- Main entry: http://localhost (Nginx reverse proxy)
- Service paths:
  - /postgres - PostgreSQL (internal)
  - /pgbouncer - PgBouncer connection pool
  - /api - PostgREST REST API
  - /pgadmin - pgAdmin web interface
  - /prometheus - Prometheus metrics
  - /grafana - Grafana dashboards
  - /redis - Redis cache (internal)
  - /docs - Sphinx documentation
  - /elasticsearch - Elasticsearch search
  - /logstash - Logstash processing (internal)
  - /kibana - Kibana visualization

Note: Individual ports are commented out in docker-compose.yml; all access is through the Nginx proxy for security.