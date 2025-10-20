Troubleshooting
===============

Common issues and solutions.

Port Conflicts
--------------

**Error**: Port already in use.

**Solution**:

.. code-block:: bash

   sudo netstat -tulpn | grep :80
   # Kill conflicting process or change ports in docker-compose.yaml

SSL Issues
----------

**Error**: SSL certificate invalid.

**Solution**:

- Regenerate certificates: `make certs`
- For production: Use Let's Encrypt.

Database Connection
-------------------

**Error**: Can't connect to database.

**Solution**:

- Check `.env` for correct DB_PASSWORD.
- Verify db service is running: `docker-compose ps`

Service Failures
----------------

**Error**: Service not starting.

**Solution**:

.. code-block:: bash

   docker-compose logs <service>
   # Fix issues and restart: docker-compose restart <service>

File Upload Limits
------------------

**Error**: File too large.

**Solution**:

- Increase `client_max_body_size` in `nginx/conf.d/default.conf`.
- Restart nginx: `docker-compose restart nginx`

Performance Issues
------------------

**Symptoms**: Slow response.

**Solutions**:

- Add more RAM.
- Optimize Redis cache.
- Scale services.

Logs and Debugging
------------------

View all logs:

.. code-block:: bash

   docker-compose logs -f

For specific service:

.. code-block:: bash

   docker-compose logs -f app

Enable debug mode in Nextcloud:

- Go to Settings > Administration > Logging.
- Set log level to Debug.

Common Errors
-------------

.. list-table:: Common Errors
   :header-rows: 1

   * - Error
     - Cause
     - Solution
   * - 502 Bad Gateway
     - App service down
     - `docker-compose restart app`
   * - 504 Timeout
     - Slow database
     - Optimize queries or add resources
   * - SSL Handshake Failed
     - Certificate issue
     - Regenerate certificates

Getting Help
------------

- Check `docker-compose ps` for service status.
- Search `Nextcloud Forums <https://help.nextcloud.com>`_.
- Open an issue on the project repository.

Preventive Measures
-------------------

- Monitor resources with `docker stats`.
- Set up alerts for service failures.
- Keep Docker and images updated.