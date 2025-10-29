Usage
=====

This section provides comprehensive guidance on using the MinIO Docker Compose setup for various scenarios.

Starting and Stopping MinIO
----------------------------

Basic Operations
~~~~~~~~~~~~~~~~

**Start MinIO Server**:

.. code-block:: bash

   make up
   # or
   docker-compose up -d

**Stop MinIO Server**:

.. code-block:: bash

   make down
   # or
   docker-compose down

**View Logs**:

.. code-block:: bash

   make logs
   # or
   docker-compose logs -f dvc-minio

**Restart Services**:

.. code-block:: bash

   make restart
   # or
   docker-compose restart

Accessing MinIO Interfaces
--------------------------

Web Console
~~~~~~~~~~~

Access the MinIO Console at ``http://localhost:30707``:

- **Username**: DVC
- **Password**: uTAntEMTuVpcJucNjOJm

The console provides:

- Bucket management
- User administration
- Access key creation
- Monitoring and metrics

S3 API Endpoint
~~~~~~~~~~~~~~~

The S3-compatible API is available at ``http://localhost:30706``.

Use any S3-compatible client:

- AWS CLI
- MinIO Client (mc)
- Boto3 (Python)
- Cyberduck, etc.

MinIO Client Setup
------------------

Install and configure the MinIO client:

.. code-block:: bash

   # Install mc
   wget https://dl.min.io/client/mc/release/linux-amd64/mc
   chmod +x mc
   sudo mv mc /usr/local/bin/

   # Configure alias
   mc alias set myminio http://localhost:30706 DVC uTAntEMTuVpcJucNjOJm

   # Test connection
   mc ls myminio/

Basic MinIO Operations
----------------------

Creating Buckets
~~~~~~~~~~~~~~~~

Via MinIO Console:

1. Log in to the web console
2. Click "Create Bucket"
3. Enter bucket name (e.g., "datasets")
4. Configure versioning and quotas as needed

Via MinIO Client:

.. code-block:: bash

   mc mb myminio/datasets

Uploading Files
~~~~~~~~~~~~~~~

Via MinIO Client:

.. code-block:: bash

   mc cp myfile.txt myminio/datasets/

Via Web Console:

1. Navigate to the bucket
2. Click "Upload"
3. Select files or drag-and-drop

Downloading Files
~~~~~~~~~~~~~~~~~

Via MinIO Client:

.. code-block:: bash

   mc cp myminio/datasets/myfile.txt .

Via Web Console:

1. Select the file
2. Click "Download"

Managing Access Policies
~~~~~~~~~~~~~~~~~~~~~~~~

Create access keys for applications:

Via Web Console:

1. Go to "Access Keys" section
2. Click "Create Access Key"
3. Set description and permissions

Via MinIO Client:

.. code-block:: bash

   mc admin user add myminio newuser newpassword
   mc admin policy set myminio readwrite user=newuser

Integration with DVC
--------------------

DVC Configuration
~~~~~~~~~~~~~~~~~

Configure DVC to use MinIO as remote storage:

.. code-block:: bash

   # Initialize DVC in your project
   dvc init

   # Add MinIO remote
   dvc remote add -d myremote s3://datasets
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm

   # Add and push data
   dvc add data.csv
   dvc push

DVC Workflow Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Track dataset
   dvc add data/train.csv
   dvc add models/model.pkl

   # Push to MinIO
   dvc push

   # Share with team
   git add .
   git commit -m "Add dataset and model"
   git push

   # Pull on another machine
   git pull
   dvc pull

Monitoring and Maintenance
--------------------------

Health Checks
~~~~~~~~~~~~~

Check MinIO health:

.. code-block:: bash

   curl http://localhost:30706/minio/health/live

View metrics via console or API.

Backup and Recovery
~~~~~~~~~~~~~~~~~~~

**Backup Data**:

.. code-block:: bash

   # Stop MinIO
   docker-compose down

   # Backup data directory
   tar -czf minio_backup.tar.gz /mnt/DVC_tmp/DVC_data

   # Restart MinIO
   docker-compose up -d

**Restore Data**:

.. code-block:: bash

   # Stop MinIO
   docker-compose down

   # Restore data
   tar -xzf minio_backup.tar.gz -C /

   # Start MinIO
   docker-compose up -d

Log Rotation
~~~~~~~~~~~~

MinIO logs are available via Docker:

.. code-block:: bash

   docker-compose logs --tail=100 dvc-minio

For persistent logging, consider external logging drivers.

Performance Optimization
------------------------

For high-performance setups:

1. **Use SSD Storage**: Mount volumes on SSD drives
2. **Network Configuration**: Use host networking for low latency
3. **Resource Allocation**: Set CPU and memory limits
4. **Erasure Coding**: Configure data protection levels

Scaling Considerations
----------------------

This setup is suitable for development and small-scale production. For larger deployments:

- Use MinIO distributed mode
- Implement load balancing
- Configure object locking for compliance
- Set up monitoring with Prometheus/Grafana