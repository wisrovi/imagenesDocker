Troubleshooting
==============

This section provides solutions to common issues you might encounter when using Docker-in-Docker.

Common Issues
-------------

Portainer Not Accessible
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Portainer web interface is not accessible on http://localhost:9003

**Solutions**:

1. **Check if Portainer container is running**:

   .. code-block:: bash

      docker-compose ps

2. **Check Portainer logs**:

   .. code-block:: bash

      docker-compose logs portainer

3. **Verify port mapping**:

   .. code-block:: bash

      docker-compose exec dind docker ps | grep portainer

4. **Restart Portainer**:

   .. code-block:: bash

      docker-compose restart dind

SSH Connection Issues
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Cannot connect to SSH on port 50422

**Solutions**:

1. **Check SSH service status**:

   .. code-block:: bash

      docker-compose exec dind ps aux | grep ssh

2. **Verify SSH port is open**:

   .. code-block:: bash

      nc -z localhost 50422

3. **Check SSH logs**:

   .. code-block:: bash

      docker-compose exec dind tail -f /var/log/auth.log

4. **Test SSH connection**:

   .. code-block:: bash

      ssh -v root@localhost -p 50422

DinD Functionality Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Docker commands fail inside the DinD container

**Solutions**:

1. **Check Docker daemon status**:

   .. code-block:: bash

      docker-compose exec dind docker info

2. **Verify privileged mode**:

   .. code-block:: bash

      docker-compose exec dind capsh --print | grep cap_sys_admin

3. **Check Docker daemon logs**:

   .. code-block:: bash

      docker-compose logs dind | grep dockerd

SSL Certificate Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: SSL certificates are not working

**Solutions**:

1. **Check certificate files**:

   .. code-block:: bash

      ls -la volumes/ssl/

2. **Verify certificate validity**:

   .. code-block:: bash

      openssl x509 -in volumes/ssl/certs/server.crt -text -noout

3. **Test SSL connection**:

   .. code-block:: bash

      curl -k https://localhost:8443

Monitoring Issues
~~~~~~~~~~~~~~~~~

**Problem**: Prometheus/Grafana not collecting metrics

**Solutions**:

1. **Check Prometheus targets**:

   .. code-block:: bash

      curl http://localhost:9090/targets

2. **Verify Grafana datasource**:

   Access Grafana at http://localhost:3000 and check datasources

3. **Check service health**:

   .. code-block:: bash

      docker-compose ps prometheus grafana

Performance Issues
~~~~~~~~~~~~~~~~~~

**Problem**: Services are running slow

**Solutions**:

1. **Check resource usage**:

   .. code-block:: bash

      docker stats

2. **Monitor system resources**:

   .. code-block:: bash

      docker-compose exec dind htop

3. **Check logs for errors**:

   .. code-block:: bash

      docker-compose logs --tail=100

Backup Issues
~~~~~~~~~~~~~

**Problem**: Automated backups are failing

**Solutions**:

1. **Check backup script permissions**:

   .. code-block:: bash

      ls -la scripts/backup.sh

2. **Verify backup directory**:

   .. code-block:: bash

      ls -la volumes/backups/

3. **Check backup logs**:

   .. code-block:: bash

      docker-compose exec dind tail -f /var/log/backup.log

Volume Persistence Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Data is not persisting between container restarts

**Solutions**:

1. **Check volume mounts**:

   .. code-block:: bash

      docker-compose exec dind df -h | grep /app

2. **Verify volume data**:

   .. code-block:: bash

      ls -la volumes/files/

3. **Check Docker volume status**:

   .. code-block:: bash

      docker volume ls

Debugging Commands
------------------

General Debugging
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View all logs
   docker-compose logs -f

   # Check service health
   docker-compose ps

   # View resource usage
   docker stats

   # Enter container shell
   docker-compose exec dind sh

Network Debugging
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check port mappings
   docker-compose ps

   # Test network connectivity
   docker-compose exec dind ping -c 3 google.com

   # Check DNS resolution
   docker-compose exec dind nslookup google.com

Security Debugging
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check firewall status
   docker-compose exec dind ufw status

   # View fail2ban status
   docker-compose exec dind fail2ban-client status

   # Check SSL certificates
   docker-compose exec dind openssl s_client -connect localhost:443

Log Analysis
~~~~~~~~~~~~

.. code-block:: bash

   # View recent logs
   docker-compose logs --tail=50

   # Follow logs in real-time
   docker-compose logs -f dind

   # Search logs for errors
   docker-compose logs | grep ERROR

Getting Help
------------

If you cannot resolve an issue using this guide:

1. Check the `GitHub Issues <https://github.com/your-repo/issues>`_ for similar problems
2. Create a new issue with:
   - Your operating system and Docker version
   - Complete error messages and logs
   - Steps to reproduce the issue
   - Your docker-compose.yml and .env file (with sensitive data removed)

System Requirements
-------------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM
- 20GB free disk space
- Linux/Windows/MacOS

Recommended Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- Docker Engine 24.0+
- Docker Compose 2.20+
- 8GB RAM
- 50GB free disk space
- Linux host system