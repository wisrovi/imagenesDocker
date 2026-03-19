API Reference
=============

This section provides an overview of MinIO's S3-compatible API and key endpoints.

S3 API Compatibility
--------------------

MinIO implements the Amazon S3 API, providing full compatibility with S3 clients and SDKs.

Base URL
~~~~~~~~

All API requests use the base URL:

.. code-block:: text

    https://localhost:30706

Authentication
~~~~~~~~~~~~~~

MinIO supports AWS Signature Version 4 authentication.

Common Operations
-----------------

Service Operations
~~~~~~~~~~~~~~~~~~

List Buckets
^^^^^^^^^^^^

**Endpoint:** ``GET /``

**Description:** Lists all buckets owned by the authenticated user.

**Response:**

.. code-block:: xml

    <ListAllMyBucketsResult>
        <Buckets>
            <Bucket>
                <Name>my-bucket</Name>
                <CreationDate>2023-01-01T00:00:00Z</CreationDate>
            </Bucket>
        </Buckets>
    </ListAllMyBucketsResult>

Bucket Operations
~~~~~~~~~~~~~~~~~

Create Bucket
^^^^^^^^^^^^^

**Endpoint:** ``PUT /bucket-name``

**Description:** Creates a new bucket.

**Headers:**
- ``x-amz-acl``: Bucket ACL (optional)

List Objects
^^^^^^^^^^^^

**Endpoint:** ``GET /bucket-name``

**Description:** Lists objects in a bucket.

**Query Parameters:**
- ``prefix``: Object key prefix
- ``delimiter``: Grouping delimiter
- ``max-keys``: Maximum number of keys to return

**Response:**

.. code-block:: xml

    <ListBucketResult>
        <Contents>
            <Key>object-key</Key>
            <Size>1234</Size>
            <LastModified>2023-01-01T00:00:00Z</LastModified>
        </Contents>
    </ListBucketResult>

Object Operations
~~~~~~~~~~~~~~~~~

Put Object
^^^^^^^^^^

**Endpoint:** ``PUT /bucket-name/object-key``

**Description:** Uploads an object to a bucket.

**Headers:**
- ``Content-Type``: MIME type of the object
- ``x-amz-acl``: Object ACL

Get Object
^^^^^^^^^^

**Endpoint:** ``GET /bucket-name/object-key``

**Description:** Downloads an object from a bucket.

**Query Parameters:**
- ``versionId``: Specific version of the object

Delete Object
^^^^^^^^^^^^^

**Endpoint:** ``DELETE /bucket-name/object-key``

**Description:** Deletes an object from a bucket.

MinIO-Specific Endpoints
------------------------

Web Console
~~~~~~~~~~~

**URL:** ``https://localhost:30707``

**Description:** Web-based management interface.

Health Check
~~~~~~~~~~~~

**Endpoint:** ``GET /minio/health/live``

**Description:** Liveness probe for load balancers.

**Response:** ``200 OK`` if healthy.

Metrics
~~~~~~~

**Endpoint:** ``GET /minio/v2/metrics/cluster``

**Description:** Prometheus-compatible metrics endpoint.

**Requires:** Metrics enabled in configuration.

Admin API
~~~~~~~~~

MinIO provides additional administrative endpoints for:

- User management
- Policy management
- Configuration management
- Server information

These endpoints are accessed via the MinIO client or SDK.

SDK Examples
------------

Python SDK Methods
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from minio import Minio

    client = Minio('localhost:30706', 'access_key', 'secret_key', secure=True)

    # Bucket operations
    client.make_bucket('my-bucket')
    client.list_buckets()

    # Object operations
    client.put_object('my-bucket', 'object-key', data, length)
    client.get_object('my-bucket', 'object-key')
    client.remove_object('my-bucket', 'object-key')

JavaScript SDK Methods
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

    const Minio = require('minio');

    const client = new Minio.Client({
        endPoint: 'localhost',
        port: 30706,
        useSSL: true,
        accessKey: 'access_key',
        secretKey: 'secret_key'
    });

    // Bucket operations
    client.makeBucket('my-bucket');
    client.listBuckets();

    // Object operations
    client.putObject('my-bucket', 'object-key', stream);
    client.getObject('my-bucket', 'object-key');
    client.removeObject('my-bucket', 'object-key');

Error Responses
---------------

MinIO returns standard S3 error responses:

.. list-table:: Common Error Codes
   :header-rows: 1
   :widths: 20 30 50

   * - Code
     - Description
     - HTTP Status
   * - AccessDenied
     - Access denied
     - 403
   * - NoSuchBucket
     - Bucket does not exist
     - 404
   * - NoSuchKey
     - Object does not exist
     - 404
   * - InvalidBucketName
     - Invalid bucket name
     - 400
   * - BucketAlreadyExists
     - Bucket already exists
     - 409

Rate Limiting
-------------

MinIO does not implement explicit rate limiting by default. However, performance may be limited by:

- Network bandwidth
- Disk I/O
- CPU resources
- Docker container limits

Versioning
----------

MinIO supports object versioning when enabled:

.. code-block:: bash

    # Enable versioning via mc
    mc version enable myminio/my-bucket

Versioned operations include:

- ``versionId`` parameter for object operations
- ``versions`` query parameter for listing operations
- Delete markers for soft deletes

Presigned URLs
--------------

Generate temporary access URLs:

.. code-block:: python

    from minio import Minio
    import datetime

    client = Minio('localhost:30706', 'access_key', 'secret_key', secure=True)

    # Generate presigned URL (expires in 1 hour)
    url = client.presigned_get_object(
        'my-bucket',
        'object-key',
        expires=datetime.timedelta(hours=1)
    )

Multipart Upload
----------------

For large files, use multipart upload:

1. Initiate multipart upload
2. Upload parts
3. Complete multipart upload

This provides better error recovery and resumable uploads.

Additional Resources
--------------------

- `MinIO API Documentation <https://docs.min.io/docs/minio-admin-complete-guide.html>`_
- `AWS S3 API Reference <https://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html>`_
- `MinIO Go SDK <https://docs.min.io/docs/golang-client-api-reference.html>`_