Quickstart Guide
================

Get Harbor up and running in minutes with this quickstart guide.

Prerequisites Check
-------------------

.. code-block:: bash

   # Check Docker
   docker --version

   # Check Docker Compose
   docker-compose --version

Download and Install
--------------------

.. tabs::

   .. tab:: Online Installation

      .. code-block:: bash

         # Download Harbor
         wget https://github.com/goharbor/harbor/releases/download/v2.14.0/harbor-online-installer-v2.14.0.tgz
         tar -xzf harbor-online-installer-v2.14.0.tgz
         cd harbor

         # Configure
         vi harbor.yml  # Set hostname and password

         # Install
         ./install.sh

   .. tab:: Offline Installation

      .. code-block:: bash

         # Download offline package
         wget https://github.com/goharbor/harbor/releases/download/v2.14.0/harbor-offline-installer-v2.14.0.tgz
         tar -xzf harbor-offline-installer-v2.14.0.tgz
         cd harbor

         # Configure
         vi harbor.yml

         # Install
         ./install.sh

Access Harbor
-------------

- **Web UI**: http://your-hostname
- **Username**: admin
- **Password**: Harbor12345 (change immediately!)

Push Your First Image
---------------------

.. code-block:: bash

   # Login to Harbor
   docker login your-hostname

   # Tag and push
   docker tag nginx:latest your-hostname/library/nginx:latest
   docker push your-hostname/library/nginx:latest

Next Steps
----------

- :doc:`installation` - Detailed installation guide
- :doc:`tutorials` - Step-by-step tutorials
- :doc:`usage` - Basic usage instructions