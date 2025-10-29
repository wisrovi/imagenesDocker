Configuration
=============

This section details the configuration options available for the MinIO Docker Compose setup.

Docker Compose Configuration
----------------------------

The ``docker-compose.yaml`` file contains the following key sections:

.. code-block:: yaml

   version: "3.8"

   services:
     dvc-minio:
       image: minio/minio:RELEASE.2025-02-28T09-55-16Z
       command: server /data --console-address ":9001"
       restart: always
       environment:
         - MINIO_ROOT_USER=DVC
         - MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm
       ports:
         - "30706:9000"
         - "30707:9001"
       volumes:
         - /mnt/DVC_tmp/DVC_data:/data

Environment Variables
---------------------

Configure MinIO behavior through environment variables:

.. list-table:: Environment Variables
   :header-rows: 1
   :widths: 25 25 50

   * - Variable
     - Default Value
     - Description
   * - MINIO_ROOT_USER
     - DVC
     - Admin username
   * - MINIO_ROOT_PASSWORD
     - uTAntEMTuVpcJucNjOJm
     - Admin password
   * - MINIO_DEFAULT_BUCKETS
     - (uncommented)
     - Comma-separated bucket list
   * - MINIO_DOMAIN
     - localhost
     - Domain for MinIO server
   * - MINIO_BROWSER
     - on
     - Enable/disable web console

.. tip::
   For production deployments, use strong, unique credentials and consider enabling TLS.

Port Configuration
------------------

The setup exposes two ports:

- **30706**: S3 API endpoint (maps to container port 9000)
- **30707**: MinIO Console (maps to container port 9001)

To change ports, modify the ``ports`` section in ``docker-compose.yaml``:

.. code-block:: yaml

   ports:
     - "9000:9000"  # S3 API
     - "9001:9001"  # Console

Volume Mounting
---------------

Data persistence is achieved through volume mounting:

.. code-block:: yaml

   volumes:
     - /mnt/DVC_tmp/DVC_data:/data

This mounts the host directory ``/mnt/DVC_tmp/DVC_data`` to the container's ``/data`` directory.

.. note::
   Ensure the host directory exists and has appropriate permissions before starting the container.

Advanced Configuration
----------------------

MinIO Server Command Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``command`` directive specifies MinIO server options:

.. code-block:: bash

   server /data --console-address ":9001"

Common options include:

- ``--address ":9000"``: Specify API server address
- ``--console-address ":9001"``: Web console address
- ``--certs-dir /certs``: Directory for TLS certificates

Restart Policies
~~~~~~~~~~~~~~~~

The ``restart: always`` policy ensures the container restarts automatically on failure or system reboot.

Other options:

- ``no``: Never restart
- ``on-failure``: Restart only on non-zero exit codes
- ``unless-stopped``: Restart unless explicitly stopped

Resource Limits
~~~~~~~~~~~~~~~

For production use, add resource constraints:

.. code-block:: yaml

   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
       reservations:
         cpus: '1.0'
         memory: 2G

Security Considerations
-----------------------

1. **Network Security**:

   - Use firewalls to restrict access to MinIO ports
   - Consider using Docker networks for isolation

2. **Data Encryption**:

   - Enable server-side encryption in MinIO
   - Use encrypted volumes for sensitive data

3. **Access Control**:

   - Implement IAM policies for fine-grained access
   - Use MinIO's built-in user management

4. **TLS Configuration**:

   For secure deployments, configure TLS:

   .. code-block:: yaml

      volumes:
        - ./certs:/root/.minio/certs
      environment:
        - MINIO_SERVER_URL=https://minio.example.com:9000

Troubleshooting Configuration Issues
------------------------------------

Common configuration problems and solutions:

1. **Port Conflicts**: Use ``netstat -tlnp | grep :30706`` to check port usage
2. **Permission Errors**: Ensure host directory permissions allow Docker access
3. **Volume Mount Issues**: Verify host path exists and is accessible
4. **Environment Variables**: Check for typos in variable names and values