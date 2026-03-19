Usage
=====

Basic Operations
----------------

Push an Image
~~~~~~~~~~~~~

.. code-block:: bash

   # Tag image for Harbor
   docker tag my-image:latest your-registry.com/library/my-image:latest

   # Login to Harbor
   docker login your-registry.com

   # Push image
   docker push your-registry.com/library/my-image:latest

Pull an Image
~~~~~~~~~~~~~

.. code-block:: bash

   docker pull your-registry.com/library/my-image:latest

Using the Scripts
-----------------

Individual Scripts
~~~~~~~~~~~~~~~~~~

- ``1.sh``: Pull image from Docker Hub
- ``2.sh``: Tag image for Harbor registry
- ``3.sh``: Push image to Harbor
- ``4.sh``: Remove local image copy
- ``5.sh``: Pull image from Harbor registry

Usage example:

.. code-block:: bash

   ./scripts/1.sh nginx latest
   ./scripts/2.sh nginx latest
   ./scripts/3.sh nginx latest
   ./scripts/4.sh nginx latest
   ./scripts/5.sh nginx latest

Bulk Upload Script
~~~~~~~~~~~~~~~~~~

``upload_multiple.sh`` automates the process for multiple images:

.. code-block:: bash

   ./scripts/upload_multiple.sh

This script processes a predefined list of popular images (Python, Node.js, Java, etc.) and uploads them to Harbor.