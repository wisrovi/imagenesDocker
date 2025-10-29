Examples
========

This section provides practical examples of using MinIO with various programming languages and tools.

Python Examples
---------------

MinIO Python SDK
~~~~~~~~~~~~~~~~

Install the MinIO Python SDK:

.. code-block:: bash

    pip install minio

Basic Connection
^^^^^^^^^^^^^^^^

.. code-block:: python

    from minio import Minio
    from minio.error import S3Error

    # Create MinIO client
    client = Minio(
        "localhost:30706",
        access_key="DVC",
        secret_key="uTAntEMTuVpcJucNjOJm",
        secure=True,  # Use HTTPS
        cert_check=False  # For self-signed certificates
    )

    print("Connected to MinIO server")

Create Bucket
^^^^^^^^^^^^^

.. code-block:: python

    try:
        # Create bucket if it doesn't exist
        if not client.bucket_exists("my-bucket"):
            client.make_bucket("my-bucket")
            print("Bucket 'my-bucket' created")
        else:
            print("Bucket 'my-bucket' already exists")
    except S3Error as exc:
        print("Error:", exc)

Upload File
^^^^^^^^^^^

.. code-block:: python

    try:
        # Upload file to bucket
        client.fput_object(
            "my-bucket",
            "my-file.txt",
            "/path/to/local/file.txt"
        )
        print("File uploaded successfully")
    except S3Error as exc:
        print("Error:", exc)

Download File
^^^^^^^^^^^^^

.. code-block:: python

    try:
        # Download file from bucket
        client.fget_object(
            "my-bucket",
            "my-file.txt",
            "/path/to/download/file.txt"
        )
        print("File downloaded successfully")
    except S3Error as exc:
        print("Error:", exc)

List Objects
^^^^^^^^^^^^

.. code-block:: python

    try:
        # List objects in bucket
        objects = client.list_objects("my-bucket")
        for obj in objects:
            print(obj.object_name, obj.size, obj.last_modified)
    except S3Error as exc:
        print("Error:", exc)

Delete Object
^^^^^^^^^^^^^

.. code-block:: python

    try:
        # Delete object from bucket
        client.remove_object("my-bucket", "my-file.txt")
        print("Object deleted successfully")
    except S3Error as exc:
        print("Error:", exc)

Boto3 Examples
~~~~~~~~~~~~~~

AWS SDK for Python (Boto3) can also be used with MinIO.

Install Boto3:

.. code-block:: bash

    pip install boto3

Configure Client
^^^^^^^^^^^^^^^^

.. code-block:: python

    import boto3
    from botocore.client import Config

    # Create S3 client for MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url='https://localhost:30706',
        aws_access_key_id='DVC',
        aws_secret_access_key='uTAntEMTuVpcJucNjOJm',
        config=Config(signature_version='s3v4'),
        verify=False  # For self-signed certificates
    )

    print("Connected to MinIO via Boto3")

Operations with Boto3
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # List buckets
    response = s3_client.list_buckets()
    for bucket in response['Buckets']:
        print(bucket['Name'])

    # Upload file
    s3_client.upload_file('/local/file.txt', 'my-bucket', 'remote-file.txt')

    # Download file
    s3_client.download_file('my-bucket', 'remote-file.txt', '/local/download.txt')

    # List objects
    response = s3_client.list_objects_v2(Bucket='my-bucket')
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'], obj['Size'])

JavaScript/Node.js Examples
---------------------------

MinIO JavaScript SDK
~~~~~~~~~~~~~~~~~~~~

Install the MinIO JavaScript SDK:

.. code-block:: bash

    npm install minio

Basic Usage
^^^^^^^^^^^

.. code-block:: javascript

    const Minio = require('minio');

    // Create MinIO client
    const minioClient = new Minio.Client({
        endPoint: 'localhost',
        port: 30706,
        useSSL: true,
        accessKey: 'DVC',
        secretKey: 'uTAntEMTuVpcJucNjOJm',
        rejectUnauthorized: false  // For self-signed certificates
    });

    console.log('Connected to MinIO');

    // Create bucket
    minioClient.makeBucket('my-bucket', 'us-east-1', (err) => {
        if (err) {
            console.log('Error creating bucket:', err);
        } else {
            console.log('Bucket created successfully');
        }
    });

    // Upload file
    const file = '/path/to/file.txt';
    minioClient.fPutObject('my-bucket', 'file.txt', file, (err, etag) => {
        if (err) {
            console.log('Error uploading file:', err);
        } else {
            console.log('File uploaded successfully');
        }
    });

AWS SDK for JavaScript
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

    const AWS = require('aws-sdk');

    // Configure AWS SDK for MinIO
    const s3 = new AWS.S3({
        endpoint: 'https://localhost:30706',
        accessKeyId: 'DVC',
        secretAccessKey: 'uTAntEMTuVpcJucNjOJm',
        s3ForcePathStyle: true,
        signatureVersion: 'v4',
        rejectUnauthorized: false
    });

    // List buckets
    s3.listBuckets((err, data) => {
        if (err) {
            console.log('Error:', err);
        } else {
            console.log('Buckets:', data.Buckets);
        }
    });

Command Line Examples
---------------------

MinIO Client (mc)
~~~~~~~~~~~~~~~~~

Install MinIO Client:

.. code-block:: bash

    # Download and install mc
    wget https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x mc
    sudo mv mc /usr/local/bin/

Configure alias:

.. code-block:: bash

    mc alias set myminio https://localhost:30706 DVC uTAntEMTuVpcJucNjOJm

Basic operations:

.. code-block:: bash

    # List buckets
    mc ls myminio

    # Create bucket
    mc mb myminio/my-bucket

    # Upload file
    mc cp myfile.txt myminio/my-bucket/

    # Download file
    mc cp myminio/my-bucket/myfile.txt .

    # List objects
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

    aws configure set default.s3.endpoint_url https://localhost:30706
    aws configure set default.s3.signature_version s3v4

Operations:

.. code-block:: bash

    # List buckets
    aws s3 ls

    # Create bucket
    aws s3 mb s3://my-bucket

    # Upload file
    aws s3 cp myfile.txt s3://my-bucket/

    # Download file
    aws s3 cp s3://my-bucket/myfile.txt .

    # Sync directory
    aws s3 sync ./local-dir s3://my-bucket/

Docker Integration
------------------

Using MinIO in Docker Compose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example docker-compose.yml for an application using MinIO:

.. code-block:: yaml

    version: '3.8'
    services:
      myapp:
        image: myapp:latest
        environment:
          - MINIO_ENDPOINT=https://minio:9000
          - MINIO_ACCESS_KEY=DVC
          - MINIO_SECRET_KEY=uTAntEMTuVpcJucNjOJm
        depends_on:
          - minio
        networks:
          - minio-network

      minio:
        image: minio/minio:latest
        command: server /data --console-address ":9001"
        environment:
          - MINIO_ROOT_USER=DVC
          - MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm
        ports:
          - "9000:9000"
          - "9001:9001"
        volumes:
          - minio_data:/data
        networks:
          - minio-network

    volumes:
      minio_data:

    networks:
      minio_network:
        driver: bridge

CI/CD Integration
-----------------

GitHub Actions Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    name: Upload to MinIO
    on: [push]
    jobs:
      upload:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
          - name: Upload to MinIO
            uses: shallwefootball/minio-action@v1
            with:
              endpoint: 'localhost:30706'
              access_key: 'DVC'
              secret_key: 'uTAntEMTuVpcJucNjOJm'
              bucket: 'my-bucket'
              file_path: './dist/*'
              insecure: true