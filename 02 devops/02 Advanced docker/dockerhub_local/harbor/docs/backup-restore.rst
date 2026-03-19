Backup and Restore
==================

Backup
------

.. code-block:: bash

   # Stop Harbor
   docker-compose down

   # Backup data directory
   tar -czf harbor-backup.tar.gz data/

   # Backup database
   docker run --rm -v harbor_data:/data -v $(pwd):/backup \
     postgres:13 pg_dump -h harbor-db -U postgres registry > backup.sql

Restore
-------

.. code-block:: bash

   # Restore data
   tar -xzf harbor-backup.tar.gz

   # Restore database
   docker run --rm -v harbor_data:/data -v $(pwd):/backup \
     postgres:13 psql -h harbor-db -U postgres registry < backup.sql

   # Start Harbor
   docker-compose up -d