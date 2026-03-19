Usage
=====

Accessing MinIO
---------------

Web Console
~~~~~~~~~~~

Access the MinIO web interface at https://localhost:30707

Login with the configured credentials.

S3 API
~~~~~~

The S3-compatible API is available at https://localhost:30706

Basic Operations
----------------

Creating Buckets
~~~~~~~~~~~~~~~~

Via Web Console:
1. Log in to the console
2. Click "Create Bucket"
3. Enter bucket name
4. Configure settings as needed

Via Command Line:

.. code-block:: bash

    # Using AWS CLI
    aws s3 mb s3://my-bucket --endpoint-url https://localhost:30706

    # Using MinIO Client
    mc mb myminio/my-bucket

Uploading Files
~~~~~~~~~~~~~~~

Web Console:
1. Select a bucket
2. Click "Upload"
3. Select files or drag and drop

Command Line:

.. code-block:: bash

    # AWS CLI
    aws s3 cp myfile.txt s3://my-bucket/ --endpoint-url https://localhost:30706

    # MinIO Client
    mc cp myfile.txt myminio/my-bucket/

Downloading Files
~~~~~~~~~~~~~~~~~

Web Console:
1. Navigate to the file
2. Click "Download"

Command Line:

.. code-block:: bash

    # AWS CLI
    aws s3 cp s3://my-bucket/myfile.txt . --endpoint-url https://localhost:30706

    # MinIO Client
    mc cp myminio/my-bucket/myfile.txt .

Managing Permissions
~~~~~~~~~~~~~~~~~~~~

Set bucket policies via the web console or API.

Common Commands
---------------

Service Management
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start MinIO
    make up

    # Stop MinIO
    make down

    # Restart MinIO
    make restart

    # View logs
    make logs

Data Management
~~~~~~~~~~~~~~~

.. code-block:: bash

    # List buckets
    aws s3 ls --endpoint-url https://localhost:30706

    # List objects in bucket
    aws s3 ls s3://my-bucket --endpoint-url https://localhost:30706

    # Remove object
    aws s3 rm s3://my-bucket/myfile.txt --endpoint-url https://localhost:30706

    # Remove bucket
    aws s3 rb s3://my-bucket --endpoint-url https://localhost:30706

User Management
~~~~~~~~~~~~~~~

Create users through the web console under "Identity" > "Users".

Best Practices
--------------

Data Organization
~~~~~~~~~~~~~~~~~

* Use meaningful bucket names
* Organize objects with prefixes (folders)
* Implement lifecycle policies for data retention

Security
~~~~~~~~

* Use HTTPS for all connections
* Implement proper access controls
* Regularly rotate credentials
* Monitor access logs

Performance
~~~~~~~~~~~

* Use appropriate object sizes
* Implement caching where possible
* Monitor performance metrics
* Scale resources as needed

Backup and Recovery
~~~~~~~~~~~~~~~~~~~

* Regular backups of data directory
* Test restore procedures
* Implement disaster recovery plans
* Use versioning for critical data

Monitoring
----------

Health Checks
~~~~~~~~~~~~~

Check MinIO health:

.. code-block:: bash

    curl -k https://localhost:30706/minio/health/live

Logs
~~~~

View application logs:

.. code-block:: bash

    docker-compose logs -f dvc-minio

Metrics
~~~~~~~

Access metrics endpoint (if enabled):

.. code-block:: bash

    curl -k https://localhost:30706/minio/v2/metrics/cluster

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Connection Refused**
    Ensure MinIO is running and ports are accessible.

**SSL Certificate Errors**
    Accept self-signed certificate or install proper certificates.

**Authentication Failed**
    Verify credentials in configuration.

**Permission Denied**
    Check user permissions and bucket policies.

**Out of Space**
    Monitor disk usage and clean up unnecessary data.

Support Resources
-----------------

* `MinIO Documentation <https://docs.min.io/>`_
* `MinIO GitHub Issues <https://github.com/minio/minio/issues>`_
* `Community Forums <https://forum.min.io/>`_