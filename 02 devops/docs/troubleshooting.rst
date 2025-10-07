Troubleshooting
===============

This section provides solutions to common issues.

General Issues
--------------

Port Conflicts
~~~~~~~~~~~~~~

**Problem**: A service fails to start due to port already in use.

**Solution**:

1. Check which process is using the port:

   .. code-block:: bash

      lsof -i :port_number

2. Stop the conflicting process or change the port mapping in ``docker-compose.yaml``

Permission Issues
~~~~~~~~~~~~~~~~~

**Problem**: Permission denied errors when accessing volumes.

**Solution**:

1. Check file permissions on the host:

   .. code-block:: bash

      ls -la /path/to/volume

2. Adjust permissions if necessary:

   .. code-block:: bash

      chown -R user:group /path/to/volume

Service-Specific Issues
-----------------------

Jenkins
~~~~~~~

**SSH Key Issues**:

- Ensure SSH keys are in ``~/.ssh`` with correct permissions (600)
- Test SSH connection: ``ssh -T git@github.com``

**Docker Socket Access**:

- Ensure the user can access ``/var/run/docker.sock``
- Check permissions: ``ls -la /var/run/docker.sock``

Kafka
~~~~~

**Connection Refused**:

- Wait for Zookeeper and Kafka to fully start (may take a few minutes)
- Check logs: ``docker-compose logs kafka``

**Topic Creation Issues**:

- Use the provided Python scripts for testing
- Ensure broker is accessible on ``localhost:9092``

SSL Certificates
~~~~~~~~~~~~~~~~

**Certificate Validation Errors**:

- For Let's Encrypt: Ensure DNS propagation
- For self-signed: Add certificates to trust store or use ``--insecure`` flag

Logs
----

View service logs:

.. code-block:: bash

   docker-compose logs -f service_name

Stop and clean up:

.. code-block:: bash

   docker-compose down -v