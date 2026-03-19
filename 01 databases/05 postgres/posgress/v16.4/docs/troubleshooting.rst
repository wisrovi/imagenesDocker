Troubleshooting
===============

Common issues and solutions.

Database Connection Issues
--------------------------

**Error: Connection refused**

- Check if PostgreSQL container is running: ``docker-compose ps``
- Verify port mapping: ``docker-compose logs postgres``

**Authentication failed**

- Check .env file credentials
- Ensure secrets are properly mounted

Service Startup Issues
----------------------

**Container fails to start**

- Check logs: ``docker-compose logs <service>``
- Verify dependencies are running

Performance Issues
------------------

**Slow queries**

- Check pg_stat_statements
- Review postgresql.conf settings
- Monitor with Prometheus

Backup/Restore Issues
---------------------

**Backup fails**

- Check disk space
- Verify database connectivity
- Review cron logs

Logs
----

View logs for all services:

.. code-block:: bash

   docker-compose logs

View specific service logs:

.. code-block:: bash

   docker-compose logs postgres