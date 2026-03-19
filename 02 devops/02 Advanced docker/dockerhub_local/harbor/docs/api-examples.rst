API Examples
============

Python examples for interacting with Harbor API.

Authentication
--------------

.. code-block:: python

   import requests
   from requests.auth import HTTPBasicAuth

   HARBOR_URL = "http://your-harbor-host"
   USERNAME = "admin"
   PASSWORD = "password"

   auth = HTTPBasicAuth(USERNAME, PASSWORD)

List Projects
-------------

.. code-block:: python

   response = requests.get(f"{HARBOR_URL}/api/v2.0/projects", auth=auth)
   projects = response.json()

   for project in projects:
       print(f"Project: {project['name']}")

List Repositories
-----------------

.. code-block:: python

   project_name = "library"
   response = requests.get(f"{HARBOR_URL}/api/v2.0/projects/{project_name}/repositories", auth=auth)
   repos = response.json()

   for repo in repos:
       print(f"Repository: {repo['name']}")

Push Image Programmatically
---------------------------

.. code-block:: python

   import docker

   client = docker.from_env()
   image = client.images.get("nginx:latest")

   # Tag and push
   image.tag(f"{HARBOR_URL}/library/nginx:latest")
   client.images.push(f"{HARBOR_URL}/library/nginx:latest", auth_config={
       'username': USERNAME,
       'password': PASSWORD
   })

Error Handling
--------------

.. code-block:: python

   try:
       response = requests.get(f"{HARBOR_URL}/api/v2.0/projects", auth=auth)
       response.raise_for_status()
       projects = response.json()
   except requests.exceptions.RequestException as e:
       print(f"Error: {e}")

Complete Example
----------------

.. code-block:: python

   # harbor_client.py
   import requests
   from requests.auth import HTTPBasicAuth

   class HarborClient:
       def __init__(self, url, username, password):
           self.url = url
           self.auth = HTTPBasicAuth(username, password)

       def get_projects(self):
           response = requests.get(f"{self.url}/api/v2.0/projects", auth=self.auth)
           return response.json()

       def create_project(self, name, public=False):
           data = {"project_name": name, "public": public}
           response = requests.post(f"{self.url}/api/v2.0/projects", json=data, auth=self.auth)
           return response.status_code == 201

   # Usage
   client = HarborClient("http://harbor.example.com", "admin", "password")
   projects = client.get_projects()
   print(projects)