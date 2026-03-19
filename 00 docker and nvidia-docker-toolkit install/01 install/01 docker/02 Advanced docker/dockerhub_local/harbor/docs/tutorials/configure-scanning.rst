Configure Vulnerability Scanning
=================================

Enable Trivy vulnerability scanning in Harbor.

Step 1: Enable Trivy
--------------------

Ensure Trivy is enabled in harbor.yml:

.. code-block:: yaml

   trivy:
     enabled: true

Step 2: Restart Harbor
----------------------

.. code-block:: bash

   docker-compose down
   docker-compose up -d

Step 3: Configure Scanning Policy
---------------------------------

1. Go to Administration > Interrogation Services
2. Configure scanning settings

Step 4: Scan an Image
---------------------

Push an image and view scan results in the Web UI.