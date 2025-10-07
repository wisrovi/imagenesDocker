Code Quality Assurance
======================

This section documents the code quality and monitoring tools available in the ``Code QA/`` directory.

SonarQube
---------

SonarQube is a code quality analysis tool that performs automatic reviews to detect bugs, code smells, and security vulnerabilities.

Location: ``Code QA/SonarQube/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/SonarQube/
   docker-compose up -d

Access: ``http://localhost:9000`` (default credentials: admin/admin)

Configuration
~~~~~~~~~~~~~

- Nginx proxy configuration in ``nginx/conf/nginx.conf``
- Persistent data storage via Docker volumes

Auto-Healing Services
---------------------

Auto-healing services monitor container health and automatically restart failed containers.

Location: ``Code QA/Check _services_status/autohealth/``

Components
~~~~~~~~~~

- **autoheal**: Main auto-healing service
- **autoheal_crontab**: Cron-based health checks

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/autohealth/autoheal/
   docker-compose up -d

Portainer
---------

Portainer provides a web UI for managing Docker containers.

Location: ``Code QA/Check _services_status/portained/``

Normal Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/portained/normal/
   docker-compose up -d

SSL Installation
~~~~~~~~~~~~~~~~

For SSL-enabled Portainer:

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/portained/ssl/
   docker-compose up -d

Access: ``http://localhost:9000`` or ``https://localhost:9443``

Uptime Kuma
-----------

Uptime Kuma is a self-hosted uptime monitoring tool.

Location: ``Code QA/Check _services_status/uptime_kuma/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/uptime_kuma/
   docker-compose up -d

Access: ``http://localhost:3001``

Dashboard Shortcuts
-------------------

Heimdal and Homer provide dashboard interfaces with shortcuts to various services.

Heimdal
~~~~~~~

Location: ``Code QA/Check _services_status/URL shortcuts/heimdall/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/URL\ shortcuts/heimdall/
   docker-compose up -d

Homer
~~~~~

Location: ``Code QA/Check _services_status/URL shortcuts/homer/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Check\ _services_status/URL\ shortcuts/homer/
   docker-compose up -d

Documentation Tools
-------------------

MediaWiki
~~~~~~~~~

A wiki engine for collaborative documentation.

Location: ``Code QA/Documentation/mediawiki/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Documentation/mediawiki/
   docker-compose up -d

Snippet Box
~~~~~~~~~~~

A simple code snippet manager.

Location: ``Code QA/Documentation/snippet-box/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Code\ QA/Documentation/snippet-box/
   docker-compose up -d