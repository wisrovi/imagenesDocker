Configuration
=============

MinIO Server Configuration
---------------------------

The main MinIO service is configured through ``docker-compose.yaml``:

.. literalinclude:: ../docker-compose.yaml
   :language: yaml
   :linenos:

Key Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

* ``MINIO_ROOT_USER``: Administrator username
* ``MINIO_ROOT_PASSWORD``: Administrator password (minimum 8 characters)
* ``MINIO_CERT_PUBLIC_KEY``: Path to public certificate file
* ``MINIO_CERT_PRIVATE_KEY``: Path to private key file

Port Configuration
^^^^^^^^^^^^^^^^^^

* ``30706:9000``: S3 API port mapping
* ``30707:9001``: Web console port mapping

Volume Mounts
^^^^^^^^^^^^^

* ``./DVC_data:/data``: Persistent data storage
* ``./certs:/root/.minio/certs``: SSL certificate directory

SSL Certificate Configuration
-----------------------------

Certificate Generation
~~~~~~~~~~~~~~~~~~~~~~~

SSL certificates are generated using the OpenSSL configuration in ``openssl/nginx/conf/openssl_wisrovi.cnf``:

.. literalinclude:: ../openssl/nginx/conf/openssl_wisrovi.cnf
   :language: ini
   :linenos:

Certificate Details
~~~~~~~~~~~~~~~~~~~

* **Algorithm**: RSA 4096-bit
* **Validity**: 825 days
* **Domain**: www.dvc.ecapturedtech.com
* **Subject Alternative Names**: https://ecapturedtech.com/

Docker Image Configuration
---------------------------

The certificate generation uses a custom Docker image defined in ``openssl/nginx/Dockerfile``:

.. literalinclude:: ../openssl/nginx/Dockerfile
   :language: dockerfile
   :linenos:

Advanced Configuration
----------------------

Custom Domain
~~~~~~~~~~~~~

To use a different domain:

1. Edit ``openssl/nginx/conf/openssl_wisrovi.cnf``
2. Update the ``CN`` and ``subjectAltName`` fields
3. Regenerate certificates

Custom Ports
~~~~~~~~~~~~

Modify port mappings in ``docker-compose.yaml``:

.. code-block:: yaml

    ports:
      - "9000:9000"  # S3 API
      - "9001:9001"  # Console

.. note::
    Ensure the new ports are not already in use.

Data Directory
~~~~~~~~~~~~~~

Change the data storage location:

.. code-block:: yaml

    volumes:
      - "/path/to/your/data:/data"

MinIO Version
~~~~~~~~~~~~~

Update to a specific MinIO version:

.. code-block:: yaml

    image: minio/minio:RELEASE.2023-01-01T00-00-00Z

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Additional MinIO configuration options:

.. code-block:: yaml

    environment:
      - MINIO_ROOT_USER=myuser
      - MINIO_ROOT_PASSWORD=mysecurepassword
      - MINIO_REGION=us-west-1
      - MINIO_BROWSER_REDIRECT_URL=https://console.example.com

Production Considerations
-------------------------

SSL Certificates
~~~~~~~~~~~~~~~~

For production deployments:

1. Use certificates from a trusted Certificate Authority
2. Implement certificate rotation policies
3. Configure HSTS headers if using a web server proxy

Security Hardening
~~~~~~~~~~~~~~~~~~

* Change default credentials
* Use strong, unique passwords
* Implement network segmentation
* Enable audit logging
* Regular security updates

Backup Strategy
~~~~~~~~~~~~~~~

* Regular backups of ``DVC_data/`` directory
* Test restore procedures
* Offsite backup storage
* Automated backup scripts

Monitoring
~~~~~~~~~~

Consider implementing:

* Health checks
* Log aggregation
* Performance monitoring
* Alerting systems

High Availability
~~~~~~~~~~~~~~~~~

For production HA setup:

* Multiple MinIO nodes
* Load balancer configuration
* Shared storage backend
* Redundancy planning