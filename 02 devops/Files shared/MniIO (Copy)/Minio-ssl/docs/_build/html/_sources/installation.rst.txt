Installation
============

System Requirements
-------------------

Before installing, ensure your system meets these requirements:

* **Operating System**: Linux, macOS, or Windows with WSL2
* **Docker**: Version 20.10 or later
* **Docker Compose**: Version 2.0 or later
* **RAM**: Minimum 2GB, recommended 4GB+
* **Disk Space**: At least 10GB for data and containers
* **Network**: Access to Docker Hub for image downloads

Installation Steps
------------------

Step 1: Download the Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone or download the MinIO SSL Setup repository:

.. code-block:: bash

    git clone <repository-url>
    cd minio-ssl-setup

Alternatively, download and extract the ZIP archive.

Step 2: Verify Docker Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure Docker and Docker Compose are properly installed:

.. code-block:: bash

    docker --version
    docker-compose --version

Step 3: Generate SSL Certificates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate the required SSL certificates:

.. tabs::

   .. tab:: Using Makefile

      .. code-block:: bash

         make certs

   .. tab:: Manual Command

      .. code-block:: bash

         cd openssl
         docker-compose up

This process will:

* Build a custom OpenSSL Docker image
* Generate a 4096-bit RSA key pair
* Create a self-signed certificate valid for 825 days
* Save certificates to the ``certs/`` directory

Step 4: Start MinIO Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch the MinIO server with SSL enabled:

.. tabs::

   .. tab:: Using Makefile

      .. code-block:: bash

         make up

   .. tab:: Manual Command

      .. code-block:: bash

         docker-compose up -d

Step 5: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check that MinIO is running correctly:

.. code-block:: bash

    docker-compose ps

You should see the ``dvc-minio`` service in "Up" status.

Access Points
-------------

After successful installation, MinIO will be accessible at:

* **Web Console**: https://localhost:30707
* **S3 API Endpoint**: https://localhost:30706

.. note::
   Since we're using self-signed certificates, your browser may show a security warning. This is normal for development environments.

Default Credentials
-------------------

* **Username**: DVC
* **Password**: uTAntEMTuVpcJucNjOJm

.. warning::
   Change these default credentials immediately in production environments!

Post-Installation Configuration
-------------------------------

1. **Change Default Password**: Update credentials in ``docker-compose.yaml``
2. **Configure Firewall**: Ensure ports 30706 and 30707 are accessible
3. **Set Up Backups**: Configure regular backups of the ``DVC_data/`` directory
4. **SSL Certificate**: For production, replace self-signed certificates with CA-signed ones

Troubleshooting Installation
-----------------------------

Common issues and solutions:

**Certificate Generation Fails**
    Ensure Docker has write permissions to the project directory.

**Port Already in Use**
    Change port mappings in ``docker-compose.yaml`` or stop conflicting services.

**Insufficient Permissions**
    Run commands with appropriate privileges or configure Docker permissions.

**Container Won't Start**
    Check logs with ``docker-compose logs`` for detailed error messages.

Next Steps
----------

Once installed, proceed to:

* :doc:`configuration` for advanced setup options
* :doc:`usage` for basic operations
* :doc:`examples` for code samples