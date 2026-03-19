Automate with Scripts
======================

Use the provided scripts for automated image management.

Using Individual Scripts
------------------------

.. code-block:: bash

   # Pull from Docker Hub
   ./scripts/1.sh nginx latest

   # Tag for Harbor
   ./scripts/2.sh nginx latest

   # Push to Harbor
   ./scripts/3.sh nginx latest

   # Remove local copy
   ./scripts/4.sh nginx latest

   # Pull from Harbor
   ./scripts/5.sh nginx latest

Bulk Upload
-----------

.. code-block:: bash

   ./scripts/upload_multiple.sh

Using Makefile
--------------

.. code-block:: bash

   # Upload images
   make upload-images

   # Pull images
   make pull-all-images