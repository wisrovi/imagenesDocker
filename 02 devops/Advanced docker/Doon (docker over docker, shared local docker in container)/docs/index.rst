Docker on Docker Documentation
===============================

Welcome to the Docker on Docker project documentation. This project provides a Docker container setup that enables running Docker commands from within a container using the "Docker out of Docker" (DooD) approach.

Overview
--------

This setup is particularly useful for development environments, CI/CD pipelines, or scenarios where you need to build and manage Docker images from within a containerized environment.

Features
--------

- **Base Image**: Ubuntu 22.04 for stability and compatibility
- **SSH Access**: Built-in SSH server for remote container access
- **Docker CLI**: Pre-installed Docker command-line interface
- **Volume Mounting**: Persistent data storage and Docker socket access
- **Timezone Configuration**: Customizable timezone settings
- **Auto-restart**: Container automatically restarts on failure

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

- Docker (version 20.10 or later recommended)
- Docker Compose (version 1.29 or later)

Installation
~~~~~~~~~~~~

1. Clone or download this repository.
2. Navigate to the project directory.
3. Run ``docker-compose up -d`` to start the container.

Usage
-----

Starting the Container
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker-compose up -d

Accessing the Container
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ssh root@localhost -p 50422

Running Docker Commands
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker ps
   docker run hello-world

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

- ``TZ``: Timezone setting (default: Europe/Madrid)

Volumes
~~~~~~~

- ``/var/run/docker.sock``: Host Docker socket
- ``./files:/app``: Data persistence

Security Considerations
-----------------------

- Change the default root password immediately.
- Be cautious with Docker socket mounting.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. Permission denied: Ensure Docker socket access.
2. SSH connection refused: Check container status.

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   configuration

Author
------

This project was created by Wisrovi Rodriguez.

For more information, visit: https://es.linkedin.com/in/wisrovi-rodriguez

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`