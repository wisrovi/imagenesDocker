API Usage
=========

Harbor provides a REST API for programmatic access. Examples:

List Projects
-------------

.. code-block:: bash

   curl -u admin:password http://your-registry.com/api/v2.0/projects

List Repositories
-----------------

.. code-block:: bash

   curl -u admin:password http://your-registry.com/api/v2.0/projects/library/repositories

List Artifacts
--------------

.. code-block:: bash

   curl -u admin:password http://your-registry.com/api/v2.0/projects/library/repositories/nginx/artifacts