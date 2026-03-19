Containers
==========

Detailed information about each container service.

PostgreSQL
----------

.. tabs::

   .. tab:: Overview

      The core database service running PostgreSQL 16.4 with custom extensions.

   .. tab:: Configuration

      - Port: 5433
      - Extensions: PostGIS, PL/Python, pg_cron, pg_buffercache
      - Custom config: ``config/postgresql.conf``

   .. tab:: Environment

      - ``POSTGRES_USER``
      - ``POSTGRES_PASSWORD``
      - ``POSTGRES_DB``

PgBouncer
---------

Connection pooling for improved performance.

PostgREST
---------

Automatic REST API generation.

pgAdmin
-------

Web-based database administration.

Prometheus
----------

Metrics collection and monitoring.

Grafana
-------

Visualization dashboards.

Flyway
------

Database schema migrations.

Backup Service
--------------

Automated backup management.

Redis
-----

In-memory caching.

Docs
----

Sphinx documentation server.