CI/CD Pipelines
===============

This section covers the CI/CD pipeline tools in the ``Pipelines/`` directory.

Jenkins
-------

Jenkins is an open-source automation server for CI/CD pipelines.

Location: ``Pipelines/Jenkins/``

Features
~~~~~~~~

- Docker integration
- Git support with SSH keys
- Python 3.10 environment
- Persistent configuration

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Pipelines/Jenkins/
   docker-compose up -d --build

Access: ``http://localhost:50443``

Configuration
~~~~~~~~~~~~~

- Custom Dockerfile for additional tools
- Volume mounts for Jenkins home, SSH keys, and Docker socket
- Port mapping: 50443:8080

Pipeline Structure
~~~~~~~~~~~~~~~~~~

The repository includes a recommended pipeline structure:

.. code-block:: text

   jenkins/
   ├── <project>/
   │   ├── build/
   │   │   ├── 1-environment_preparation/
   │   │   ├── 2-db_config/
   │   │   └── ...
   │   ├── deploy/
   │   ├── test/
   │   ├── QA/
   │   ├── PR reviewer/
   │   └── train/

Each subdirectory contains Jenkins pipeline files (``.jenkinsfile``) for different environments (DEVELOPMENT, TEST, PRODUCTION).

n8n
---

n8n is a workflow automation tool for technical and business users.

Location: ``Pipelines/n8n/``

Normal Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd Pipelines/n8n/
   docker-compose -f docker-compose.normal.yaml up -d

SSL Installation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd Pipelines/n8n/
   docker-compose -f docker-compose.ssl.yaml up -d

Access: ``http://localhost:5678`` or ``https://localhost:5678``

Configuration
~~~~~~~~~~~~~

- SQLite database for data persistence
- Environment variables for customization
- Volume mounts for data and configuration