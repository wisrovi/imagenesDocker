Monitoring and Dashboards
==========================

This section documents monitoring tools and dashboards.

Central Logging
---------------

Centralized logging service for aggregating logs from multiple sources.

Location: ``central_logs/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd central_logs/
   docker-compose up -d

Heimdal Dashboard
-----------------

Heimdal is a dashboard for organizing links to various services.

Location: ``heimdall/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd heimdall/
   docker-compose up -d

Access: ``http://localhost:80``

Configuration
~~~~~~~~~~~~~

- SQLite database for configuration
- Customizable tiles and links
- Nginx proxy with PHP support