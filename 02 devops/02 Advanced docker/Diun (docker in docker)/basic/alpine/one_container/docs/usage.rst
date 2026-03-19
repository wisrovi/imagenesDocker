Usage
=====

Starting the Environment
------------------------

::

    docker-compose up -d

Accessing Services
------------------

1. **Portainer Web Interface**:

   - URL: `http://localhost:50421`
   - Default credentials: `admin` / `admin`
   - First login will prompt you to set a new password

2. **SSH Access**::

    ssh root@localhost -p 50422

   - Password: As set in `.env` (default: `password`)

3. **Web Terminal**:

   - URL: `http://localhost:50423`
   - Provides a browser-based shell interface

Docker Operations Inside the Container
---------------------------------------

Once inside the container (via SSH or web terminal), you can run Docker commands::

    # Check Docker version
    docker --version

    # Run a test container
    docker run hello-world

    # List running containers
    docker ps

    # Build and run your own containers
    docker build -t my-app .
    docker run -d my-app

Managing the Environment
------------------------

::

    # View logs
    docker-compose logs -f

    # Access container shell
    docker-compose exec dind-basic sh

    # Stop services
    docker-compose down

    # Stop and remove volumes
    docker-compose down -v

    # Rebuild the image
    docker-compose build --no-cache