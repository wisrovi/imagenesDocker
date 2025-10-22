Database
========

PostgreSQL configuration and usage.

Connecting
----------

.. tabs::

   .. tab:: Direct

      .. code-block:: bash

         psql -h localhost -p 5433 -U perseus -d perseus

   .. tab:: Via PgBouncer

      .. code-block:: bash

         psql -h localhost -p 6432 -U perseus -d perseus

Extensions
----------

- **PostGIS**: Geographic data support
- **PL/Python**: Python stored procedures
- **pg_cron**: Job scheduling
- **pg_buffercache**: Buffer cache inspection
- **pg_stat_statements**: Query statistics

Configuration
-------------

Custom settings in ``config/postgresql.conf``:

- Shared buffers: 256MB
- Work mem: 4MB
- Maintenance work mem: 64MB
- WAL level: replica