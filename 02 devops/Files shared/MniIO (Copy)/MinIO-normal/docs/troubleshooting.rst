Troubleshooting
===============

This section provides solutions to common issues encountered with the MinIO Docker Compose setup.

Container Issues
----------------

MinIO Container Won't Start
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Container exits immediately or fails to start.

**Possible Causes and Solutions**:

1. **Port Conflicts**:

   .. code-block:: bash

      # Check if ports are in use
      netstat -tlnp | grep :30706
      netstat -tlnp | grep :30707

   **Solution**: Change ports in ``docker-compose.yaml`` or free up the ports.

2. **Volume Permission Issues**:

   .. code-block:: bash

      # Check directory permissions
      ls -la /mnt/DVC_tmp/DVC_data

   **Solution**:

   .. code-block:: bash

      sudo chown -R 1000:1000 /mnt/DVC_tmp/DVC_data
      sudo chmod -R 755 /mnt/DVC_tmp/DVC_data

3. **Insufficient Resources**:

   **Solution**: Ensure adequate CPU and memory are available. Add resource limits to ``docker-compose.yaml``:

   .. code-block:: yaml

      deploy:
        resources:
          limits:
            memory: 2G
            cpus: '1.0'

4. **Check Logs**:

   .. code-block:: bash

      docker-compose logs dvc-minio

Container Runs But MinIO Unavailable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Container is running but cannot access MinIO console or API.

**Solutions**:

1. **Check Container Health**:

   .. code-block:: bash

      docker-compose ps
      curl http://localhost:30706/minio/health/live

2. **Verify Port Mapping**:

   .. code-block:: bash

      docker port dvc-minio

3. **Check Firewall**:

   .. code-block:: bash

      sudo ufw status
      # Allow ports if needed
      sudo ufw allow 30706
      sudo ufw allow 30707

Connection Issues
-----------------

Cannot Connect to MinIO from Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Clients cannot connect to MinIO endpoints.

**Troubleshooting Steps**:

1. **Verify Endpoints**:

   .. code-block:: bash

      # Test API endpoint
      curl -I http://localhost:30706

      # Test console endpoint
      curl -I http://localhost:30707

2. **Check Network Configuration**:

   - Ensure Docker is using the correct network mode
   - Verify that the host can reach the container IP

3. **DNS Resolution**:

   If using custom domains, check DNS resolution:

   .. code-block:: bash

      nslookup localhost

Authentication Problems
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Login fails or access denied errors.

**Solutions**:

1. **Verify Credentials**:

   Check that the credentials in client configuration match those in ``docker-compose.yaml``.

2. **Check Signature Version**:

   Ensure clients use Signature Version 4:

   .. code-block:: python

      from botocore.client import Config
      config = Config(signature_version='s3v4')

3. **Region Configuration**:

   MinIO requires a region to be specified:

   .. code-block:: python

      client = boto3.client(
          's3',
          region_name='us-east-1',  # Required for MinIO
          # ... other config
      )

Data Management Issues
----------------------

File Upload/Download Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common Issues**:

1. **File Size Limits**:

   MinIO has default limits. For large files, use multipart upload:

   .. code-block:: python

      # For files > 100MB
      s3_client.create_multipart_upload(Bucket='bucket', Key='large_file')
      # ... upload parts ...

2. **Permission Errors**:

   Ensure the MinIO user has write permissions to the bucket.

3. **Network Timeouts**:

   Increase timeout settings:

   .. code-block:: python

      config = Config(
          read_timeout=300,
          connect_timeout=60
      )

Bucket Creation Errors
~~~~~~~~~~~~~~~~~~~~~~

**Error**: BucketAlreadyExists

**Solution**: Choose a unique bucket name or check existing buckets:

.. code-block:: bash

   mc ls myminio/

**Error**: InvalidBucketName

**Solution**: Bucket names must follow S3 naming conventions:

- 3-63 characters long
- Lowercase letters, numbers, hyphens only
- Must start and end with letter or number

DVC Integration Problems
------------------------

DVC Cannot Connect to MinIO
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ``dvc push`` or ``dvc pull`` fails.

**Solutions**:

1. **Verify DVC Remote Configuration**:

   .. code-block:: bash

      dvc remote list
      dvc remote show myremote

2. **Check Credentials**:

   Ensure the access key and secret match MinIO configuration.

3. **Test Connection**:

   .. code-block:: bash

      # Test with mc
      mc ls myminio/datasets/

4. **Endpoint URL**:

   Make sure the endpoint URL includes the protocol:

   .. code-block:: bash

      dvc remote modify myremote endpointurl http://localhost:30706

DVC Push/Pull Errors
~~~~~~~~~~~~~~~~~~~~

**Common Issues**:

1. **Large Files**:

   For large files, DVC may need special configuration:

   .. code-block:: bash

      dvc remote modify myremote region us-east-1
      dvc remote modify myremote use_ssl false

2. **Path Issues**:

   Ensure DVC cache and workspace paths are correct.

3. **Version Conflicts**:

   Update DVC to the latest version:

   .. code-block:: bash

      pip install --upgrade dvc

Performance Issues
------------------

Slow Uploads/Downloads
~~~~~~~~~~~~~~~~~~~~~~

**Optimization Tips**:

1. **Use SSD Storage**: Ensure data directory is on SSD.

2. **Network Configuration**: Use host networking for better performance:

   .. code-block:: yaml

      network_mode: host

3. **Connection Pooling**: Configure client connection pools:

   .. code-block:: python

      config = Config(
          max_pool_connections=20,
          retries={'max_attempts': 3}
      )

4. **Multipart Uploads**: Use for files > 100MB.

High Memory Usage
~~~~~~~~~~~~~~~~~

**Solutions**:

1. **Set Memory Limits**:

   .. code-block:: yaml

      deploy:
        resources:
          limits:
            memory: 4G

2. **Monitor Usage**:

   .. code-block:: bash

      docker stats dvc-minio

3. **Tune MinIO Settings**:

   Environment variables for memory management:

   - ``MINIO_CACHE_SIZE``: Control cache size
   - ``GOGC``: Go garbage collector tuning

Storage Issues
--------------

Disk Space Problems
~~~~~~~~~~~~~~~~~~~

**Symptoms**: Uploads fail due to insufficient space.

**Solutions**:

1. **Check Disk Usage**:

   .. code-block:: bash

      df -h /mnt/DVC_tmp/DVC_data

2. **Clean Up Old Data**:

   .. code-block:: bash

      # List large objects
      mc find myminio/ --size +100MB

      # Remove old versions
      mc rm --versions myminio/bucket/old_file

3. **Expand Storage**: Add more disk space or move to larger volume.

Permission Errors on Host Volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Container cannot write to mounted volume.

**Solutions**:

1. **Fix Permissions**:

   .. code-block:: bash

      sudo chown -R 1000:1000 /mnt/DVC_tmp/DVC_data
      sudo chmod -R 755 /mnt/DVC_tmp/DVC_data

2. **Use Named Volumes**:

   .. code-block:: yaml

      volumes:
        - minio_data:/data

      volumes:
        minio_data:

3. **Check SELinux/AppArmor**:

   Disable or configure security policies if interfering.

Backup and Recovery
-------------------

Data Corruption Issues
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Files become corrupted or inaccessible.

**Recovery Steps**:

1. **Check MinIO Logs**:

   .. code-block:: bash

      docker-compose logs dvc-minio | grep -i error

2. **Run Healing**:

   MinIO can repair corrupted data:

   .. code-block:: bash

      mc admin heal myminio/

3. **Restore from Backup**:

   If healing fails, restore from backup.

Lost Data Scenarios
~~~~~~~~~~~~~~~~~~~

**Prevention**:

1. **Enable Versioning**:

   .. code-block:: bash

      mc version enable myminio/bucket

2. **Regular Backups**:

   .. code-block:: bash

      # Create backup script
      #!/bin/bash
      TIMESTAMP=$(date +%Y%m%d_%H%M%S)
      docker-compose down
      tar -czf "minio_backup_${TIMESTAMP}.tar.gz" /mnt/DVC_tmp/DVC_data
      docker-compose up -d

3. **Use Replication**: Set up MinIO server-side replication.

Monitoring and Logging
----------------------

Enable Detailed Logging
~~~~~~~~~~~~~~~~~~~~~~~

**MinIO Logging**:

.. code-block:: yaml

   environment:
     - MINIO_LOG_LEVEL=DEBUG

**Docker Logging**:

.. code-block:: bash

   docker-compose logs -f --tail=100 dvc-minio

Set Up Monitoring
~~~~~~~~~~~~~~~~~

1. **Health Checks**:

   .. code-block:: bash

      #!/bin/bash
      if curl -f http://localhost:30706/minio/health/live; then
          echo "MinIO is healthy"
      else
          echo "MinIO is unhealthy"
          # Send alert
      fi

2. **Metrics Collection**:

   MinIO exposes Prometheus metrics at ``/minio/v2/metrics/cluster``.

Getting Help
------------

If issues persist:

1. **Check Official Documentation**:

   - `MinIO Troubleshooting Guide <https://docs.min.io/minio/baremetal/troubleshooting/troubleshooting.html>`_
   - `DVC Issues <https://github.com/iterative/dvc/issues>`_

2. **Community Support**:

   - MinIO Slack: https://slack.min.io/
   - DVC Discord: https://discord.gg/bTm8vAj

3. **Log Analysis**:

   Provide detailed logs and configuration when seeking help.

4. **System Information**:

   Include output of:

   .. code-block:: bash

      docker --version
      docker-compose --version
      uname -a
      df -h