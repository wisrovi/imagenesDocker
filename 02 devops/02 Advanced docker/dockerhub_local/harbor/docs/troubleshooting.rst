Troubleshooting
===============

Common Issues
-------------

Harbor Won't Start
~~~~~~~~~~~~~~~~~~~

- Check Docker and Docker Compose versions
- Verify hostname configuration
- Ensure ports are not in use
- Check disk space

Cannot Push Images
~~~~~~~~~~~~~~~~~~

- Verify user permissions
- Check project settings
- Ensure correct image tagging
- Validate network connectivity

Database Connection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Check PostgreSQL container status
- Verify database credentials in ``harbor.yml``
- Ensure data volume permissions

Logs and Debugging
------------------

Enable debug logging in ``harbor.yml``:

.. code-block:: yaml

   log:
     level: debug

View service logs:

.. code-block:: bash

   docker logs harbor-log