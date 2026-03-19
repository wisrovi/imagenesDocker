Installation
============

System Requirements
-------------------

Before installing MinIO Docker Setups, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows with Docker Desktop
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **CPU**: At least 1 CPU core
- **Memory**: Minimum 512 MB RAM (2 GB recommended)
- **Disk Space**: At least 1 GB for MinIO binaries and initial data
- **Network**: Access to ports 30706 and 30707

Installation Steps
------------------

1. Clone or Download the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-repo/minio-docker-setups.git
   cd minio-docker-setups

2. Choose Your Setup
~~~~~~~~~~~~~~~~~~~~~

Navigate to the desired setup directory:

.. tabs::

   .. tab:: MinIO-Normal

      .. code-block:: bash

         cd MinIO-normal

   .. tab:: MinIO-SSL

      .. code-block:: bash

         cd Minio-ssl

MinIO-Normal Installation
-------------------------

1. Configure Data Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``docker-compose.yaml`` to set the data volume path:

.. code-block:: yaml

   volumes:
     - /path/to/your/data:/data

Or use a relative path:

.. code-block:: yaml

   volumes:
     - ./DVC_data:/data

2. Start MinIO
~~~~~~~~~~~~~~

.. code-block:: bash

   docker-compose up -d

3. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~

Check that containers are running:

.. code-block:: bash

   docker-compose ps

Access the console at http://localhost:30707

MinIO-SSL Installation
----------------------

1. Generate SSL Certificates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd openssl
   docker-compose up

This creates certificates in the ``certs`` directory.

2. Start MinIO
~~~~~~~~~~~~~~

.. code-block:: bash

   cd ..
   docker-compose up -d

3. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~

Access the console at https://localhost:30707 (accept self-signed certificate)

Configuration Options
---------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Customize the deployment using environment variables in ``docker-compose.yaml``:

.. code-block:: yaml

   environment:
     - MINIO_ROOT_USER=your_username
     - MINIO_ROOT_PASSWORD=your_secure_password
     - MINIO_DEFAULT_BUCKETS=my-bucket

Port Configuration
~~~~~~~~~~~~~~~~~~

Modify ports if conflicts exist:

.. code-block:: yaml

   ports:
     - "9000:9000"  # S3 API
     - "9001:9001"  # Console

Volume Mounting
~~~~~~~~~~~~~~~

Ensure host directories exist and have proper permissions:

.. code-block:: bash

   mkdir -p /path/to/data
   chmod 755 /path/to/data

Troubleshooting Installation
----------------------------

Common Issues
~~~~~~~~~~~~~

- **Port Already in Use**: Change port mappings in ``docker-compose.yaml``
- **Permission Denied**: Ensure Docker has access to mounted directories
- **Certificate Errors**: For SSL setup, regenerate certificates if needed

Logs
~~~~

View logs for debugging:

.. code-block:: bash

   docker-compose logs -f

Uninstallation
--------------

To remove the setup:

.. code-block:: bash

   docker-compose down -v  # Remove containers and volumes

Next Steps
----------

- :doc:`usage`: Learn how to use MinIO
- :doc:`examples`: See practical examples