Troubleshooting
===============

This section covers common issues and their solutions when working with MinIO Docker Setups.

Installation Issues
-------------------

Port Conflicts
~~~~~~~~~~~~~~

**Problem**: Docker Compose fails with "port already in use" error.

**Solution**:

1. Check which service is using the ports:

   .. code-block:: bash

      lsof -i :30706
      lsof -i :30707

2. Stop the conflicting service or change ports in ``docker-compose.yaml``:

   .. code-block:: yaml

      ports:
        - "9000:9000"  # Changed from 30706
        - "9001:9001"  # Changed from 30707

3. Restart MinIO:

   .. code-block:: bash

      docker-compose up -d

Permission Issues
~~~~~~~~~~~~~~~~~

**Problem**: MinIO cannot write to the mounted volume.

**Solution**:

1. Ensure the host directory exists and has correct permissions:

   .. code-block:: bash

      mkdir -p /path/to/data
      chmod 755 /path/to/data

2. For Docker Desktop on Windows/Mac, ensure the directory is shared in Docker settings.

3. Check Docker user permissions:

   .. code-block:: bash

      ls -la /path/to/data

Disk Space Issues
~~~~~~~~~~~~~~~~~

**Problem**: MinIO fails to start due to insufficient disk space.

**Solution**:

1. Check available disk space:

   .. code-block:: bash

      df -h

2. Free up space or change the data directory to a drive with more space.

3. Monitor MinIO disk usage:

   .. code-block:: bash

      docker-compose exec dvc-minio df -h /data

SSL/TLS Issues
--------------

Certificate Generation Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: OpenSSL certificate generation fails.

**Solution**:

1. Ensure Docker has access to create files in the project directory.

2. Check OpenSSL container logs:

   .. code-block:: bash

      cd Minio-ssl/openssl
      docker-compose logs

3. Verify the OpenSSL configuration file exists and is valid.

4. Regenerate certificates:

   .. code-block:: bash

      docker-compose down
      rm -rf ../../certs
      docker-compose up

Certificate Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Browsers or clients reject the self-signed certificate.

**Solution**:

1. For development, accept the security warning in your browser.

2. Add the certificate to your system's trust store (not recommended for production).

3. For production, use a CA-signed certificate.

Connection Refused
~~~~~~~~~~~~~~~~~~

**Problem**: Cannot connect to MinIO over HTTPS.

**Solution**:

1. Ensure MinIO is running with SSL:

   .. code-block:: bash

      docker-compose ps

2. Check that certificate files exist:

   .. code-block:: bash

      ls -la certs/

3. Verify MinIO environment variables for certificate paths.

4. Check MinIO logs:

   .. code-block:: bash

      docker-compose logs dvc-minio

Runtime Issues
--------------

MinIO Not Starting
~~~~~~~~~~~~~~~~~~

**Problem**: MinIO container exits immediately.

**Solution**:

1. Check container logs:

   .. code-block:: bash

      docker-compose logs dvc-minio

2. Common causes:
   - Invalid environment variables
   - Insufficient memory
   - Corrupted data directory

3. Try starting without persisted data:

   .. code-block:: bash

      docker-compose down -v
      docker-compose up -d

High Memory Usage
~~~~~~~~~~~~~~~~~

**Problem**: MinIO consumes excessive memory.

**Solution**:

1. Set memory limits in Docker Compose:

   .. code-block:: yaml

      services:
        dvc-minio:
          deploy:
            resources:
              limits:
                memory: 1G
              reservations:
                memory: 512M

2. Monitor memory usage:

   .. code-block:: bash

      docker stats

Slow Performance
~~~~~~~~~~~~~~~~

**Problem**: MinIO operations are slow.

**Solution**:

1. Use SSD storage for data directory.

2. Ensure sufficient RAM (at least 2GB recommended).

3. Check network bandwidth.

4. Optimize Docker settings:

   .. code-block:: yaml

      services:
        dvc-minio:
          environment:
            - MINIO_REGION=us-east-1
          ulimits:
            nofile:
              soft: 65536
              hard: 65536

Client Connection Issues
------------------------

AWS CLI Errors
~~~~~~~~~~~~~~

**Problem**: AWS CLI cannot connect to MinIO.

**Solution**:

1. Verify endpoint configuration:

   .. code-block:: bash

      aws configure list

2. Ensure signature version is set to v4:

   .. code-block:: bash

      aws configure set default.s3.signature_version s3v4

3. Test connection:

   .. code-block:: bash

      aws s3 ls --endpoint-url http://localhost:30706

Python boto3 Errors
~~~~~~~~~~~~~~~~~~~

**Problem**: boto3 client fails to connect.

**Solution**:

1. Check endpoint URL and credentials:

   .. code-block:: python

      import boto3
      client = boto3.client(
          's3',
          endpoint_url='http://localhost:30706',
          aws_access_key_id='DVC',
          aws_secret_access_key='uTAntEMTuVpcJucNjOJm'
      )
      client.list_buckets()

2. Handle SSL verification for self-signed certificates:

   .. code-block:: python

      import boto3
      client = boto3.client(
          's3',
          endpoint_url='https://localhost:30706',
          aws_access_key_id='DVC',
          aws_secret_access_key='uTAntEMTuVpcJucNjOJm',
          verify=False  # Only for development
      )

MinIO Client (mc) Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: mc commands fail.

**Solution**:

1. Verify alias configuration:

   .. code-block:: bash

      mc alias list

2. Test connection:

   .. code-block:: bash

      mc admin info myminio

3. Check for SSL issues (use ``https://`` for SSL setup).

DVC Integration Issues
----------------------

Remote Configuration Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: DVC cannot connect to MinIO remote.

**Solution**:

1. Verify DVC remote configuration:

   .. code-block:: bash

      dvc remote list

2. Check remote settings:

   .. code-block:: bash

      dvc remote modify myremote endpointurl
      dvc remote modify myremote access_key_id
      dvc remote modify myremote secret_access_key

3. Test connection:

   .. code-block:: bash

      dvc remote modify myremote region us-east-1
      dvc push  # Test with a small file

Push/Pull Failures
~~~~~~~~~~~~~~~~~~

**Problem**: DVC push or pull operations fail.

**Solution**:

1. Ensure the bucket exists in MinIO.

2. Check file permissions and sizes.

3. Verify network connectivity.

4. Use verbose logging:

   .. code-block:: bash

      dvc push -v

Data Corruption Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: DVC reports corrupted data.

**Solution**:

1. Check MinIO data integrity:

   .. code-block:: bash

      mc admin heal myminio/my-bucket/

2. Re-upload corrupted files.

3. Verify local and remote file hashes.

Monitoring and Logging
----------------------

Viewing Logs
~~~~~~~~~~~~

**Problem**: Need to debug MinIO operations.

**Solution**:

1. View real-time logs:

   .. code-block:: bash

      docker-compose logs -f dvc-minio

2. Filter logs by time:

   .. code-block:: bash

      docker-compose logs --since "1h" dvc-minio

3. Export logs for analysis:

   .. code-block:: bash

      docker-compose logs dvc-minio > minio_logs.txt

Health Checks
~~~~~~~~~~~~~

**Problem**: Need to verify MinIO health.

**Solution**:

1. Check MinIO health endpoints:

   .. code-block:: bash

      curl http://localhost:30706/minio/health/live
      curl http://localhost:30706/minio/health/ready

2. Use MinIO client:

   .. code-block:: bash

      mc admin info myminio

3. Check Docker container health:

   .. code-block:: bash

      docker-compose ps

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Need to monitor MinIO performance.

**Solution**:

1. Access metrics endpoint:

   .. code-block:: bash

      curl http://localhost:30706/minio/v2/metrics/cluster

2. Use Prometheus-compatible monitoring.

3. Monitor disk I/O and network usage:

   .. code-block:: bash

      docker stats

Backup and Recovery
-------------------

Data Loss Scenarios
~~~~~~~~~~~~~~~~~~~

**Problem**: Data is lost or corrupted.

**Solution**:

1. Restore from backup if available.

2. Use MinIO's healing feature for erasure-coded data:

   .. code-block:: bash

      mc admin heal myminio/

3. For non-erasure-coded setups, restore from snapshots.

Volume Issues
~~~~~~~~~~~~~

**Problem**: Docker volume becomes inaccessible.

**Solution**:

1. Check volume status:

   .. code-block:: bash

      docker volume ls
      docker volume inspect minio_dvc_data

2. Recreate volume if necessary:

   .. code-block:: bash

      docker-compose down -v
      docker-compose up -d

Migration Issues
~~~~~~~~~~~~~~~~

**Problem**: Need to migrate data between setups.

**Solution**:

1. Use mc mirror for data migration:

   .. code-block:: bash

      mc mirror old-minio/bucket new-minio/bucket

2. For large datasets, use distributed copying tools.

3. Verify data integrity after migration:

   .. code-block:: bash

      mc find old-minio/bucket --exec "mc stat {}" | wc -l
      mc find new-minio/bucket --exec "mc stat {}" | wc -l

Getting Help
------------

If you cannot resolve an issue:

1. Check the `MinIO documentation <https://docs.min.io/>`_.
2. Search `MinIO GitHub issues <https://github.com/minio/minio/issues>`_.
3. Post on the `MinIO Slack community <https://slack.min.io/>`_.
4. For DVC issues, check `DVC documentation <https://dvc.org/doc>`_.

Diagnostic Information
~~~~~~~~~~~~~~~~~~~~~~

When reporting issues, include:

- MinIO version: ``docker-compose exec dvc-minio minio --version``
- Docker version: ``docker --version``
- Docker Compose version: ``docker-compose --version``
- Operating system and version
- Full error logs
- Configuration files (with sensitive data redacted)