Installation
============

Step 1: Download Harbor
-----------------------

Download the Harbor installer package from the official releases:

.. code-block:: bash

   # Download from GitHub releases
   wget https://github.com/goharbor/harbor/releases/download/v2.14.0/harbor-online-installer-v2.14.0.tgz
   tar -xzf harbor-online-installer-v2.14.0.tgz
   cd harbor

Step 2: Configure Harbor
------------------------

Edit the ``harbor.yml`` configuration file:

.. code-block:: yaml

   hostname: your-domain.com
   http:
     port: 80
   harbor_admin_password: YourSecurePassword
   database:
     password: YourDBPassword

Key configuration options:

- ``hostname``: External hostname for Harbor (required)
- ``http.port``: HTTP port (default: 80)
- ``harbor_admin_password``: Initial admin password
- ``data_volume``: Directory for persistent data (default: ./data)

Step 3: Run Installation
------------------------

Execute the installation script:

.. code-block:: bash

   ./install.sh

This script will:

1. Check Docker and Docker Compose installation
2. Load Harbor images (if offline package used)
3. Prepare configuration files
4. Start all Harbor services using Docker Compose

Step 4: Access Harbor
---------------------

Once installation completes:

- Web UI: http://your-domain.com
- Default admin credentials: admin / Harbor12345 (change immediately)