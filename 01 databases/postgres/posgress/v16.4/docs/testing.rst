Testing
=======

Testing strategies and tools.

Integration Tests
-----------------

Run comprehensive tests:

.. code-block:: bash

   docker-compose -f docker-compose.test.yml up --abort-on-container-exit

Unit Tests
----------

Test individual components.

Load Testing
------------

Use tools like Apache Bench or JMeter for performance testing.

Database Tests
--------------

- Test migrations with Flyway
- Verify data integrity
- Check backup/restore functionality

API Tests
---------

- Test PostgREST endpoints
- Validate responses
- Check authentication

Monitoring Tests
----------------

- Verify Prometheus metrics collection
- Test Grafana dashboards
- Check alerting rules