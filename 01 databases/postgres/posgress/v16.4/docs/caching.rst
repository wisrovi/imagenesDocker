Caching
=======

Redis for high-performance data caching.

Usage
-----

.. code-block:: python

   import redis

   r = redis.Redis(host='localhost', port=6379, db=0)
   r.set('key', 'value')
   value = r.get('key')

Use Cases
---------

- Session storage
- Query result caching
- Message queuing
- Rate limiting

Configuration
-------------

- Port: 6379
- Persistence: Append-only file
- Volume: redis_data