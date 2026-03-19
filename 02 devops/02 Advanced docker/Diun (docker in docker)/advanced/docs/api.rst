API Reference
============

The Docker-in-Docker project provides a REST API for managing containers, images, volumes, and system resources.

API Endpoints
-------------

Health Check
~~~~~~~~~~~~

.. http:get:: /api/health

   Get the health status of the API service.

   **Example request:**

   .. sourcecode:: http

      GET /api/health HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "status": "healthy",
        "service": "dind-api",
        "version": "1.0.0"
      }

Container Management
~~~~~~~~~~~~~~~~~~~

List Containers
^^^^^^^^^^^^^^^

.. http:get:: /api/containers

   Get a list of all containers.

   **Example request:**

   .. sourcecode:: http

      GET /api/containers HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "containers": [
          {
            "ID": "abc123...",
            "Names": ["my-container"],
            "Image": "nginx:latest",
            "Status": "Up 2 hours"
          }
        ],
        "count": 1
      }

Get Container Details
^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/containers/(container_id)

   Get detailed information about a specific container.

   :param container_id: The ID or name of the container

   **Example request:**

   .. sourcecode:: http

      GET /api/containers/abc123 HTTP/1.1
      Host: localhost:5000

Start Container
^^^^^^^^^^^^^^^

.. http:post:: /api/containers/(container_id)/start

   Start a stopped container.

   :param container_id: The ID or name of the container

   **Example request:**

   .. sourcecode:: http

      POST /api/containers/abc123/start HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "message": "Container abc123 started"
      }

Stop Container
^^^^^^^^^^^^^^

.. http:post:: /api/containers/(container_id)/stop

   Stop a running container.

   :param container_id: The ID or name of the container

   **Example request:**

   .. sourcecode:: http

      POST /api/containers/abc123/stop HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "message": "Container abc123 stopped"
      }

Get Container Logs
^^^^^^^^^^^^^^^^^^

.. http:get:: /api/containers/(container_id)/logs

   Get the logs of a container.

   :param container_id: The ID or name of the container
   :query lines: Number of log lines to return (default: 100)

   **Example request:**

   .. sourcecode:: http

      GET /api/containers/abc123/logs?lines=50 HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "container_id": "abc123",
        "logs": "2023-10-21 12:00:00 nginx started..."
      }

Image Management
~~~~~~~~~~~~~~~~

List Images
^^^^^^^^^^^

.. http:get:: /api/images

   Get a list of all Docker images.

   **Example request:**

   .. sourcecode:: http

      GET /api/images HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "images": [
          {
            "Repository": "nginx",
            "Tag": "latest",
            "ID": "abc123...",
            "Size": "123MB"
          }
        ],
        "count": 1
      }

Volume Management
~~~~~~~~~~~~~~~~~

List Volumes
^^^^^^^^^^^^

.. http:get:: /api/volumes

   Get a list of all Docker volumes.

   **Example request:**

   .. sourcecode:: http

      GET /api/volumes HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "volumes": [
          {
            "Name": "my-volume",
            "Driver": "local",
            "Mountpoint": "/var/lib/docker/volumes/my-volume"
          }
        ],
        "count": 1
      }

System Information
~~~~~~~~~~~~~~~~~~

System Info
^^^^^^^^^^^

.. http:get:: /api/system/info

   Get Docker system information.

   **Example request:**

   .. sourcecode:: http

      GET /api/system/info HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "ServerVersion": "24.0.6",
        "OperatingSystem": "Alpine Linux v3.18",
        "Architecture": "x86_64"
      }

System Disk Usage
^^^^^^^^^^^^^^^^^

.. http:get:: /api/system/df

   Get Docker system disk usage information.

   **Example request:**

   .. sourcecode:: http

      GET /api/system/df HTTP/1.1
      Host: localhost:5000

Backup Operations
~~~~~~~~~~~~~~~~~

Create Backup
^^^^^^^^^^^^^

.. http:post:: /api/backup

   Create a backup of all volumes.

   **Example request:**

   .. sourcecode:: http

      POST /api/backup HTTP/1.1
      Host: localhost:5000

   **Example response:**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "message": "Backup created successfully"
      }

Error Responses
---------------

All API endpoints may return the following error responses:

**404 Not Found**

.. sourcecode:: http

   HTTP/1.1 404 Not Found
   Content-Type: application/json

   {
     "error": "Endpoint not found"
   }

**500 Internal Server Error**

.. sourcecode:: http

   HTTP/1.1 500 Internal Server Error
   Content-Type: application/json

   {
     "error": "Internal server error"
   }

Authentication
--------------

The API currently does not require authentication. In production environments,
consider implementing authentication mechanisms such as:

- API keys
- JWT tokens
- OAuth2
- Basic authentication

Rate Limiting
-------------

The API implements rate limiting to prevent abuse:

- General endpoints: 100 requests per minute
- API endpoints: 10 requests per second

CORS Support
------------

The API supports Cross-Origin Resource Sharing (CORS) for web applications.

SDKs and Libraries
------------------

Python Client
~~~~~~~~~~~~~

.. code-block:: python

   import requests

   class DindAPI:
       def __init__(self, base_url="http://localhost:5000"):
           self.base_url = base_url

       def list_containers(self):
           response = requests.get(f"{self.base_url}/api/containers")
           return response.json()

       def start_container(self, container_id):
           response = requests.post(f"{self.base_url}/api/containers/{container_id}/start")
           return response.json()

   # Usage
   api = DindAPI()
   containers = api.list_containers()
   print(containers)

JavaScript Client
~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   class DindAPI {
       constructor(baseURL = 'http://localhost:5000') {
           this.baseURL = baseURL;
       }

       async listContainers() {
           const response = await fetch(`${this.baseURL}/api/containers`);
           return await response.json();
       }

       async startContainer(containerId) {
           const response = await fetch(`${this.baseURL}/api/containers/${containerId}/start`, {
               method: 'POST'
           });
           return await response.json();
       }
   }

   // Usage
   const api = new DindAPI();
   api.listContainers().then(containers => console.log(containers));

Webhooks
--------

The API supports webhooks for real-time notifications of container events.
Configure webhook URLs in the environment variables to receive notifications
when containers start, stop, or encounter errors.