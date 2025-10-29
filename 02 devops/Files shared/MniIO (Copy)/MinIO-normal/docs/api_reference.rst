API Reference
=============

This section provides reference information for MinIO APIs and client libraries.

MinIO S3 API
------------

MinIO implements the Amazon S3 API, providing compatibility with existing S3 tools and libraries.

Core Operations
~~~~~~~~~~~~~~~

**Service Operations**

- ``GET /`` - List buckets
- ``HEAD /`` - Check service availability

**Bucket Operations**

- ``GET /{bucket}`` - List objects in bucket
- ``PUT /{bucket}`` - Create bucket
- ``DELETE /{bucket}`` - Delete bucket
- ``HEAD /{bucket}`` - Check bucket existence

**Object Operations**

- ``GET /{bucket}/{object}`` - Download object
- ``PUT /{bucket}/{object}`` - Upload object
- ``DELETE /{bucket}/{object}`` - Delete object
- ``HEAD /{bucket}/{object}`` - Get object metadata
- ``POST /{bucket}/{object}`` - Complete multipart upload

Authentication
~~~~~~~~~~~~~~

MinIO uses AWS Signature Version 4 for authentication:

.. code-block:: python

   import hmac
   import hashlib
   import datetime

   def sign(key, msg):
       return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

   def get_signature_key(key, date_stamp, region_name, service_name):
       k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
       k_region = sign(k_date, region_name)
       k_service = sign(k_region, service_name)
       k_signing = sign(k_service, 'aws4_request')
       return k_signing

Python Client Libraries
-----------------------

Boto3
~~~~~

Official AWS SDK for Python, fully compatible with MinIO:

.. code-block:: python

   import boto3
   from botocore.client import Config

   # Create client
   client = boto3.client(
       's3',
       endpoint_url='http://localhost:30706',
       aws_access_key_id='DVC',
       aws_secret_access_key='uTAntEMTuVpcJucNjOJm',
       config=Config(signature_version='s3v4')
   )

   # List buckets
   response = client.list_buckets()

   # Upload file
   client.upload_file('file.txt', 'bucket', 'file.txt')

MinIO Python SDK
~~~~~~~~~~~~~~~~

Official MinIO SDK for Python:

.. code-block:: python

   from minio import Minio

   # Create client
   client = Minio(
       'localhost:30706',
       access_key='DVC',
       secret_key='uTAntEMTuVpcJucNjOJm',
       secure=False
   )

   # List buckets
   buckets = client.list_buckets()

   # Upload file
   client.fput_object('bucket', 'file.txt', 'file.txt')

Command Line Tools
------------------

MinIO Client (mc)
~~~~~~~~~~~~~~~~~

Official MinIO command-line tool:

.. code-block:: bash

   # Configure alias
   mc alias set myminio http://localhost:30706 DVC uTAntEMTuVpcJucNjOJm

   # List buckets
   mc ls myminio/

   # Create bucket
   mc mb myminio/bucket

   # Upload file
   mc cp file.txt myminio/bucket/

   # Download file
   mc cp myminio/bucket/file.txt .

AWS CLI
~~~~~~~

Amazon's command-line tool with MinIO support:

.. code-block:: bash

   # Configure profile
   aws configure --profile minio
   # AWS Access Key ID: DVC
   # AWS Secret Access Key: uTAntEMTuVpcJucNjOJm
   # Default region name: us-east-1
   # Default output format: json

   # Set endpoint
   aws configure set endpoint_url http://localhost:30706 --profile minio

   # List buckets
   aws s3 ls --profile minio

   # Upload file
   aws s3 cp file.txt s3://bucket/file.txt --profile minio

DVC API
-------

DVC Remote Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import dvc.api

   # Get data URL
   url = dvc.api.get_url('data.csv', 'data.csv')

   # Read parameters
   params = dvc.api.params_show()

   # Read metrics
   metrics = dvc.api.metrics_show()

DVC Command Line
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Initialize DVC
   dvc init

   # Add remote
   dvc remote add -d myremote s3://bucket
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm

   # Track file
   dvc add data.csv

   # Push to remote
   dvc push

   # Pull from remote
   dvc pull

REST API Endpoints
------------------

MinIO Console API
~~~~~~~~~~~~~~~~~

- ``GET /api/v1/login`` - User authentication
- ``GET /api/v1/buckets`` - List buckets
- ``POST /api/v1/buckets`` - Create bucket
- ``GET /api/v1/buckets/{bucket}/objects`` - List objects

Admin API
~~~~~~~~~

- ``GET /minio/admin/v3/info`` - Server information
- ``GET /minio/admin/v3/heal`` - Healing status
- ``POST /minio/admin/v3/service`` - Service operations

Health Check Endpoints
~~~~~~~~~~~~~~~~~~~~~~

- ``GET /minio/health/live`` - Liveness probe
- ``GET /minio/health/ready`` - Readiness probe
- ``GET /minio/health/cluster`` - Cluster health

Error Codes
-----------

Common S3 Error Responses
~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------+--------------------------------+
| Error Code         | Description                    |
+====================+================================+
| AccessDenied       | Access denied                  |
+--------------------+--------------------------------+
| BucketAlreadyExists| Bucket name already in use     |
+--------------------+--------------------------------+
| BucketNotEmpty     | Bucket is not empty            |
+--------------------+--------------------------------+
| InvalidBucketName  | Invalid bucket name            |
+--------------------+--------------------------------+
| NoSuchBucket       | Bucket does not exist          |
+--------------------+--------------------------------+
| NoSuchKey          | Object does not exist          |
+--------------------+--------------------------------+
| SignatureDoesNotMatch| Authentication failure       |
+--------------------+--------------------------------+

MinIO-Specific Errors
~~~~~~~~~~~~~~~~~~~~~

- ``InvalidRequest``: Malformed request
- ``InternalError``: Server internal error
- ``SlowDown``: Too many requests
- ``ServiceUnavailable``: Service temporarily unavailable

Rate Limiting
-------------

MinIO implements rate limiting to prevent abuse:

- Default: 5000 requests per second per IP
- Configurable via ``MINIO_RATE_LIMIT`` environment variable
- Burst allowance for temporary spikes

For high-throughput applications, consider:

- Connection pooling
- Exponential backoff for retries
- Load balancing across multiple MinIO instances