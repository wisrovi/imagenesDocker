Push and Pull Images Tutorial
==============================

This tutorial shows how to push and pull container images using Harbor.

Prerequisites
-------------

- Harbor is installed and running
- Docker client installed
- Access to Harbor registry

Step 1: Login to Harbor
-----------------------

.. code-block:: bash

   docker login your-harbor-hostname

Enter your Harbor username and password when prompted.

Step 2: Pull an Image from Docker Hub
--------------------------------------

.. code-block:: bash

   docker pull nginx:latest

Step 3: Tag the Image for Harbor
---------------------------------

.. code-block:: bash

   docker tag nginx:latest your-harbor-hostname/library/nginx:latest

Step 4: Push to Harbor
----------------------

.. code-block:: bash

   docker push your-harbor-hostname/library/nginx:latest

Step 5: Pull from Harbor
------------------------

.. code-block:: bash

   docker pull your-harbor-hostname/library/nginx:latest

Verification
------------

Check in the Harbor Web UI that the image appears in the library project.

Next Steps
----------

- :doc:`setup-replication` - Set up image replication
- :doc:`configure-scanning` - Enable vulnerability scanning