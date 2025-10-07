Docker-in-Docker
================

This section covers Docker-in-Docker configurations in the ``Docker_over_docker/`` directory.

Docker in Docker
----------------

Running Docker containers inside Docker containers.

Location: ``Docker_over_docker/docker_in_docker/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Docker_over_docker/docker_in_docker/
   docker-compose up -d

Use Case
~~~~~~~~

Useful for CI/CD pipelines that need to build Docker images.

Docker on Docker
----------------

Alternative Docker-in-Docker setup.

Location: ``Docker_over_docker/docker_on_docker/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Docker_over_docker/docker_on_docker/
   docker-compose up -d

Custom Dockerfile
~~~~~~~~~~~~~~~~~

Includes a custom Dockerfile for specialized Docker environments.