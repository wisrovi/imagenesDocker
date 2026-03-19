Usage
=====

Starting and Stopping MinIO
----------------------------

Starting MinIO
~~~~~~~~~~~~~~

.. tabs::

   .. tab:: MinIO-Normal

      .. code-block:: bash

         cd MinIO-normal
         docker-compose up -d

   .. tab:: MinIO-SSL

      .. code-block:: bash

         cd Minio-ssl
         docker-compose up -d

Stopping MinIO
~~~~~~~~~~~~~~

.. code-block:: bash

   docker-compose down

Accessing the MinIO Console
---------------------------

The MinIO Console provides a web-based interface for managing your storage.

.. tabs::

   .. tab:: MinIO-Normal

      - URL: http://localhost:30707
      - Username: DVC
      - Password: uTAntEMTuVpcJucNjOJm

   .. tab:: MinIO-SSL

      - URL: https://localhost:30707
      - Username: DVC
      - Password: uTAntEMTuVpcJucNjOJm
      - Note: Accept the self-signed certificate warning

Console Features
~~~~~~~~~~~~~~~~

- Create and manage buckets
- Upload and download objects
- Manage users and policies
- View access logs
- Configure notifications

Using MinIO with S3-Compatible Tools
-------------------------------------

MinIO Client (mc)
~~~~~~~~~~~~~~~~~

Install MinIO Client and configure:

.. code-block:: bash

   mc alias set myminio http://localhost:30706 DVC uTAntEMTuVpcJucNjOJm

Basic operations:

.. code-block:: bash

   mc mb myminio/my-bucket
   mc cp file.txt myminio/my-bucket/
   mc ls myminio/my-bucket

AWS CLI
~~~~~~~

Configure AWS CLI for MinIO:

.. code-block:: bash

   aws configure
   # AWS Access Key ID: DVC
   # AWS Secret Access Key: uTAntEMTuVpcJucNjOJm
   # Default region name: us-east-1
   # Default output format: json

Set endpoint:

.. code-block:: bash

   aws configure set default.s3.endpoint_url http://localhost:30706
   aws configure set default.s3.signature_version s3v4

Operations:

.. code-block:: bash

   aws s3 mb s3://my-bucket
   aws s3 cp file.txt s3://my-bucket/
   aws s3 ls s3://my-bucket/

Python boto3
~~~~~~~~~~~~

Install boto3:

.. code-block:: bash

   pip install boto3

Example script:

.. code-block:: python

   import boto3

   # Configure client
   s3_client = boto3.client(
       's3',
       endpoint_url='http://localhost:30706',
       aws_access_key_id='DVC',
       aws_secret_access_key='uTAntEMTuVpcJucNjOJm'
   )

   # Create bucket
   s3_client.create_bucket(Bucket='my-bucket')

   # Upload file
   s3_client.upload_file('local_file.txt', 'my-bucket', 'remote_file.txt')

   # List objects
   response = s3_client.list_objects_v2(Bucket='my-bucket')
   for obj in response.get('Contents', []):
       print(obj['Key'])

Integration with DVC
---------------------

DVC (Data Version Control) is a tool for versioning data and models in ML projects.

Configure DVC Remote
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install dvc

Add MinIO as a remote:

.. code-block:: bash

   dvc remote add -d myremote s3://my-bucket
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm

Use in Workflows
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Track a file
   dvc add data.csv

   # Push to remote
   dvc push

   # Pull from remote
   dvc pull

Managing Buckets and Objects
-----------------------------

Creating Buckets
~~~~~~~~~~~~~~~~

Via Console:
1. Log in to the MinIO Console
2. Click "Create Bucket"
3. Enter bucket name and configure settings

Via CLI:

.. code-block:: bash

   mc mb myminio/my-bucket

Uploading Objects
~~~~~~~~~~~~~~~~~

Via Console:
1. Select a bucket
2. Click "Upload"
3. Choose files or drag and drop

Via CLI:

.. code-block:: bash

   mc cp file.txt myminio/my-bucket/

Setting Permissions
~~~~~~~~~~~~~~~~~~~

MinIO supports IAM policies for fine-grained access control.

Example policy for read-only access:

.. code-block:: json

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject"],
         "Resource": ["arn:aws:s3:::my-bucket/*"]
       }
     ]
   }

Monitoring and Logs
-------------------

Viewing Logs
~~~~~~~~~~~~

.. code-block:: bash

   docker-compose logs -f dvc-minio

MinIO Metrics
~~~~~~~~~~~~~

MinIO provides Prometheus-compatible metrics. Access at ``/minio/v2/metrics/cluster``.

Health Checks
~~~~~~~~~~~~~

Check MinIO health:

.. code-block:: bash

   curl http://localhost:30706/minio/health/live

Backup and Recovery
-------------------

Data Persistence
~~~~~~~~~~~~~~~~

Data is stored in the mounted volume. To backup:

.. code-block:: bash

   docker-compose down
   cp -r /path/to/data /path/to/backup
   docker-compose up -d

Disaster Recovery
~~~~~~~~~~~~~~~~~

For distributed setups, MinIO supports erasure coding for data protection.

Next Steps
----------

- :doc:`examples`: See more detailed examples
- :doc:`api_reference`: Learn about MinIO APIs
- :doc:`troubleshooting`: Solve common issues