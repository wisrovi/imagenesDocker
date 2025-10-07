Usage
=====

Starting the Services
---------------------

.. code-block:: bash

   make start
   # or
   docker-compose up -d

Starting Individual Services
----------------------------

.. code-block:: bash

   # Start only the first Ollama instance
   make start-ollama1
   # or
   docker-compose up -d ollama_1

Accessing the Web Interface
---------------------------

- **Local Access**: http://localhost:11435
- **Remote Access**: Check tunnel logs for Cloudflare URL

  .. code-block:: bash

     docker-compose logs tunnel

Managing Models
---------------

Connect to an Ollama container to manage models:

.. code-block:: bash

   docker-compose exec ollama_1 bash
   ollama pull qwen2.5-coder
   ollama list
   ollama run qwen2.5-coder

Viewing Logs
------------

.. code-block:: bash

   make logs
   # or
   docker-compose logs -f

Stopping Services
-----------------

.. code-block:: bash

   make stop
   # or
   docker-compose down

Cleaning Up
-----------

Remove containers and volumes:

.. code-block:: bash

   make clean
   # or
   docker-compose down -v