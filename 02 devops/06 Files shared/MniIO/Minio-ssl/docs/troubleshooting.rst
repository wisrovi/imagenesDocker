Troubleshooting
===============

This section covers common issues and their solutions when working with MinIO SSL Setup.

Installation Issues
-------------------

Certificate Generation Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Certificate generation command exits with error
- No files created in ``certs/`` directory

**Solutions:**

1. Check Docker permissions:

   .. code-block:: bash

       docker run hello-world

2. Ensure write permissions on project directory:

   .. code-block:: bash

       ls -la
       chmod 755 .

3. Check available disk space:

   .. code-block:: bash

       df -h

4. Verify OpenSSL configuration:

   .. code-block:: bash

       cat openssl/nginx/conf/openssl_wisrovi.cnf

MinIO Won't Start
~~~~~~~~~~~~~~~~~

**Symptoms:**
- ``docker-compose ps`` shows container not running
- Error messages in logs

**Solutions:**

1. Check logs:

   .. code-block:: bash

       docker-compose logs dvc-minio

2. Verify certificate files exist:

   .. code-block:: bash

       ls -la certs/

3. Check port availability:

   .. code-block:: bash

       netstat -tlnp | grep 3070

4. Validate docker-compose.yaml syntax:

   .. code-block:: bash

       docker-compose config

Port Conflicts
~~~~~~~~~~~~~~

**Symptoms:**
- "Port already in use" error

**Solutions:**

1. Change ports in docker-compose.yaml:

   .. code-block:: yaml

       ports:
         - "9000:9000"  # Change from 30706
         - "9001:9001"  # Change from 30707

2. Find process using ports:

   .. code-block:: bash

       lsof -i :30706
       lsof -i :30707

3. Stop conflicting services or change ports

SSL/TLS Issues
--------------

Certificate Not Trusted
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Browser shows security warning
- API calls fail with certificate errors

**Solutions:**

1. For development: Accept self-signed certificate in browser

2. For production: Install CA-signed certificates

3. Configure client to skip verification (not recommended for production):

   .. code-block:: python

       import ssl
       ssl._create_default_https_context = ssl._create_unverified_context

Connection Refused
~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Unable to connect to MinIO endpoints
- Timeout errors

**Solutions:**

1. Verify MinIO is running:

   .. code-block:: bash

       docker-compose ps

2. Check firewall settings:

   .. code-block:: bash

       sudo ufw status
       sudo iptables -L

3. Test local connectivity:

   .. code-block:: bash

       curl -k https://localhost:30706/minio/health/live

4. Check Docker network:

   .. code-block:: bash

       docker network ls
       docker network inspect minio-ssl_default

Authentication Problems
-----------------------

Invalid Credentials
~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Login fails with "Invalid credentials" error

**Solutions:**

1. Verify credentials in docker-compose.yaml:

   .. code-block:: yaml

       environment:
         - MINIO_ROOT_USER=DVC
         - MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm

2. Check for special characters in password

3. Ensure credentials match between client and server

4. Reset credentials if necessary

Access Denied
~~~~~~~~~~~~~

**Symptoms:**
- Operations fail with permission errors

**Solutions:**

1. Check bucket policies in MinIO console

2. Verify user permissions

3. Ensure correct access keys are used

4. Check IAM policies if applicable

Data Management Issues
----------------------

Out of Disk Space
~~~~~~~~~~~~~~~~~

**Symptoms:**
- Upload fails with disk space errors
- MinIO becomes unresponsive

**Solutions:**

1. Check disk usage:

   .. code-block:: bash

       df -h
       du -sh DVC_data/

2. Clean up old data:

   .. code-block:: bash

       # Remove old objects via MinIO console or API

3. Increase disk space or add volumes

4. Implement data lifecycle policies

Corrupted Data
~~~~~~~~~~~~~~

**Symptoms:**
- Files cannot be accessed or are corrupted

**Solutions:**

1. Check data integrity:

   .. code-block:: bash

       # Use MinIO's audit logs
       docker-compose logs dvc-minio | grep -i error

2. Restore from backup if available

3. Use MinIO's healing feature:

   .. code-block:: bash

       # Access MinIO console and check healing status

Performance Issues
------------------

Slow Uploads/Downloads
~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Transfer speeds are slower than expected

**Solutions:**

1. Check network bandwidth:

   .. code-block:: bash

       speedtest-cli

2. Optimize chunk size for uploads

3. Use parallel uploads/downloads

4. Check system resources:

   .. code-block:: bash

       top
       iostat -x 1

High CPU/Memory Usage
~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- MinIO consumes excessive resources

**Solutions:**

1. Monitor resource usage:

   .. code-block:: bash

       docker stats

2. Adjust Docker resource limits:

   .. code-block:: yaml

       services:
         dvc-minio:
           deploy:
             resources:
               limits:
                 cpus: '1.0'
                 memory: 1G

3. Optimize MinIO configuration

4. Scale horizontally if needed

Logging and Monitoring
----------------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~~

For detailed troubleshooting:

.. code-block:: yaml

    environment:
      - MINIO_ROOT_USER=DVC
      - MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm
      - MINIO_LOG_LEVEL=DEBUG

Access Logs
~~~~~~~~~~~

View MinIO logs:

.. code-block:: bash

    # Real-time logs
    docker-compose logs -f dvc-minio

    # Last 100 lines
    docker-compose logs --tail=100 dvc-minio

    # Save logs to file
    docker-compose logs dvc-minio > minio_logs.txt

Health Checks
~~~~~~~~~~~~~

Monitor MinIO health:

.. code-block:: bash

    # Liveness probe
    curl -k https://localhost:30706/minio/health/live

    # Readiness probe
    curl -k https://localhost:30706/minio/health/ready

Common Error Codes
------------------

.. list-table:: Common HTTP Error Codes
   :header-rows: 1
   :widths: 20 30 50

   * - Code
     - Description
     - Solution
   * - 403
     - Access Denied
     - Check permissions/policies
   * - 404
     - Not Found
     - Verify bucket/object exists
   * - 500
     - Internal Server Error
     - Check server logs
   * - 503
     - Service Unavailable
     - Check server status

Getting Help
------------

If you cannot resolve an issue:

1. Check MinIO documentation: https://docs.min.io/
2. Search GitHub issues: https://github.com/minio/minio/issues
3. Post on community forums: https://forum.min.io/
4. Provide detailed logs and configuration when seeking help