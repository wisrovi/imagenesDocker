Usage
=====

Basic Operations
----------------

::

   # Start the services
   docker-compose up -d

   # View logs
   docker-compose logs -f

   # Access container shell
   docker-compose exec dind-basic bash

   # Stop services
   docker-compose down

   # Rebuild and restart
   docker-compose up -d --build

Using Docker Inside the Container
----------------------------------

Once inside the container (via SSH or web terminal)::

   # Check Docker status
   docker info

   # Run a test container
   docker run hello-world

   # List containers
   docker ps -a

Managing Portainer
------------------

Access Portainer at http://localhost:50421 to:

- View and manage containers
- Pull and manage images
- Configure networks and volumes
- Monitor resource usage