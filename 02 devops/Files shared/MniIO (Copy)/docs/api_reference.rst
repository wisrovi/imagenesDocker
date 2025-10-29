API Reference
=============

This section provides reference information for MinIO APIs and configurations used in the Docker setups.

MinIO S3 API
------------

MinIO implements the Amazon S3 API. For complete API documentation, see the `official MinIO S3 API documentation <https://docs.min.io/docs/minio-admin-complete-guide.html>`_.

Common Endpoints
~~~~~~~~~~~~~~~~

- **Service Operations**:
  - GET / (ListBuckets)

- **Bucket Operations**:
  - GET /bucket (ListObjects)
  - PUT /bucket (CreateBucket)
  - DELETE /bucket (DeleteBucket)

- **Object Operations**:
  - GET /bucket/key (GetObject)
  - PUT /bucket/key (PutObject)
  - DELETE /bucket/key (DeleteObject)

Authentication
~~~~~~~~~~~~~

MinIO uses AWS Signature Version 4 for authentication. The default credentials are:

- Access Key: ``DVC``
- Secret Key: ``uTAntEMTuVpcJucNjOJm``

Docker Compose Configuration
----------------------------

MinIO-Normal Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
         # - MINIO_DEFAULT_BUCKETS=datasets
       ports:
         - "30706:9000"  # S3 API
         - "30707:9001"  # Console
       volumes:
         - /mnt/DVC_tmp/DVC_data:/data

MinIO-SSL Configuration
~~~~~~~~~~~~~~~~~~~~~~~

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
         - MINIO_CERT_PUBLIC_KEY=/root/.minio/certs/public.crt
         - MINIO_CERT_PRIVATE_KEY=/root/.minio/certs/private.key
         # - MINIO_DEFAULT_BUCKETS=datasets
       ports:
         - "30706:9000"  # S3 API (HTTPS)
         - "30707:9001"  # Console (HTTPS)
       volumes:
         - ./DVC_data:/data
         - ./certs:/root/.minio/certs

Environment Variables
---------------------

MinIO Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: MinIO Environment Variables
   :header-rows: 1
   :widths: 25 50 25

   * - Variable
     - Description
     - Default
   * - MINIO_ROOT_USER
     - Root username for MinIO
     - minioadmin
   * - MINIO_ROOT_PASSWORD
     - Root password for MinIO
     - minioadmin
   * - MINIO_DEFAULT_BUCKETS
     - Comma-separated list of buckets to create on startup
     - (none)
   * - MINIO_CERT_PUBLIC_KEY
     - Path to public certificate file
     - (none)
   * - MINIO_CERT_PRIVATE_KEY
     - Path to private key file
     - (none)
   * - MINIO_SERVER_URL
     - External URL for MinIO server
     - (auto-detected)
   * - MINIO_BROWSER_REDIRECT_URL
     - External URL for MinIO Console
     - (auto-detected)

SSL Configuration
~~~~~~~~~~~~~~~~~

For SSL setup, certificates are generated using OpenSSL with the following configuration:

.. code-block:: ini

   [req]
   distinguished_name = req_distinguished_name
   req_extensions = v3_req
   prompt = no

   [req_distinguished_name]
   C = ES
   ST = Extremadura
   L = Badajoz
   O = eCaptureDtech
   OU = AI Solutions
   CN = www.dvc.ecapturedtech.com

   [v3_req]
   keyUsage = keyEncipherment, dataEncipherment
   extendedKeyUsage = serverAuth
   subjectAltName = @alt_names

   [alt_names]
   DNS.1 = www.dvc.ecapturedtech.com
   DNS.2 = dvc.ecapturedtech.com

MinIO Client (mc) Commands
--------------------------

Common mc Commands
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Configure alias
   mc alias set myminio http://localhost:30706 ACCESS_KEY SECRET_KEY

   # List buckets
   mc ls myminio/

   # Create bucket
   mc mb myminio/my-bucket

   # Upload file
   mc cp file.txt myminio/my-bucket/

   # Download file
   mc cp myminio/my-bucket/file.txt .

   # Sync directory
   mc mirror local-dir myminio/my-bucket/

   # Remove object
   mc rm myminio/my-bucket/file.txt

   # Get bucket info
   mc stat myminio/my-bucket

AWS CLI Configuration
---------------------

For AWS CLI compatibility:

.. code-block:: bash

   aws configure set default.s3.endpoint_url http://localhost:30706
   aws configure set default.s3.signature_version s3v4
   aws configure set default.s3.addressing_style path

Common AWS CLI Commands
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List buckets
   aws s3 ls

   # Create bucket
   aws s3 mb s3://my-bucket

   # Upload file
   aws s3 cp file.txt s3://my-bucket/

   # Download file
   aws s3 cp s3://my-bucket/file.txt .

   # Sync directory
   aws s3 sync local-dir s3://my-bucket/

   # Remove object
   aws s3 rm s3://my-bucket/file.txt

Python boto3 API
----------------

Client Initialization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boto3

   s3_client = boto3.client(
       's3',
       endpoint_url='http://localhost:30706',
       aws_access_key_id='DVC',
       aws_secret_access_key='uTAntEMTuVpcJucNjOJm',
       region_name='us-east-1'
   )

Common Methods
~~~~~~~~~~~~~~

.. code-block:: python

   # List buckets
   response = s3_client.list_buckets()

   # Create bucket
   s3_client.create_bucket(Bucket='my-bucket')

   # Upload file
   s3_client.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

   # Download file
   s3_client.download_file('my-bucket', 'remote_file.txt', 'local_file.txt')

   # List objects
   response = s3_client.list_objects_v2(Bucket='my-bucket')

   # Delete object
   s3_client.delete_object(Bucket='my-bucket', Key='remote_file.txt')

   # Generate presigned URL
   url = s3_client.generate_presigned_url(
       'get_object',
       Params={'Bucket': 'my-bucket', 'Key': 'remote_file.txt'},
       ExpiresIn=3600
   )

DVC Configuration
-----------------

DVC Remote Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   dvc remote add -d myremote s3://my-bucket
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm

DVC Commands
~~~~~~~~~~~~

.. code-block:: bash

   # Initialize DVC
   dvc init

   # Track file
   dvc add data.csv

   # Push to remote
   dvc push

   # Pull from remote
   dvc pull

   # List tracked files
   dvc list .

   # Show data status
   dvc status

Health Check Endpoints
----------------------

MinIO provides several health check endpoints:

- **Live Check**: ``GET /minio/health/live``
- **Ready Check**: ``GET /minio/health/ready``
- **Cluster Health**: ``GET /minio/v2/health/cluster``

Example:

.. code-block:: bash

   curl http://localhost:30706/minio/health/live

Metrics Endpoints
~~~~~~~~~~~~~~~~~

- **Cluster Metrics**: ``GET /minio/v2/metrics/cluster``
- **Node Metrics**: ``GET /minio/v2/metrics/node``

These endpoints provide Prometheus-compatible metrics for monitoring.

Configuration Files
-------------------

OpenSSL Configuration (SSL Setup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``openssl_wisrovi.cnf`` file used for certificate generation:

.. literalinclude:: ../Minio-ssl/openssl/nginx/conf/openssl_wisrovi.cnf
   :language: ini

Nginx Configuration (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced SSL termination, you can use Nginx as a reverse proxy:

.. code-block:: nginx

   server {
       listen 443 ssl;
       server_name minio.example.com;

       ssl_certificate /etc/ssl/certs/minio.crt;
       ssl_certificate_key /etc/ssl/private/minio.key;

       location / {
           proxy_pass http://localhost:30706;
           proxy_set_header Host $http_host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }