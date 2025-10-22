Quick Start
===========

Get up and running in minutes.

Prerequisites
-------------

- Docker 20.10+
- Docker Compose 1.29+

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd postgres-docker-setup

2. Configure environment:

   .. code-block:: bash

      cp .env.example .env
      # Edit .env with your settings

3. Start the services:

   .. code-block:: bash

      docker-compose up -d

4. Access the services:

   - Database: ``psql -h localhost -p 5433 -U perseus -d perseus``
   - Admin: http://localhost:5050
   - API: http://localhost:3001
   - Monitoring: http://localhost:9090

Next Steps
----------

- :doc:`setup` for detailed configuration
- :doc:`architecture` for system overview
- :doc:`api` for REST API usage