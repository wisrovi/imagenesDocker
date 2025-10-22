Configuration
=============

Environment and service configuration.

Environment Variables
---------------------

.. list-table:: Environment Variables
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - POSTGRES_USER
     - Database user
     - perseus
   * - POSTGRES_PASSWORD
     - Database password
     - (required)
   * - POSTGRES_DB
     - Database name
     - perseus

Docker Compose Overrides
------------------------

Create ``docker-compose.override.yml`` for environment-specific settings.

.. code-block:: yaml

   version: '3.8'
   services:
     postgres:
       environment:
         - POSTGRES_PASSWORD=dev_password

Volumes
-------

- postgres_data: Database files
- grafana_data: Grafana configs
- pgadmin_data: pgAdmin data
- redis_data: Redis data