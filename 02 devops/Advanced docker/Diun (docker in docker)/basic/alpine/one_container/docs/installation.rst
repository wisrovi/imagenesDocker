Installation
============

1. **Clone or download this repository**::

    git clone <repository-url>
    cd docker_in_docker/basic

2. **Copy the environment configuration**::

    cp .env.example .env

3. **Build and start the services**::

    docker-compose up -d --build

This command will:

- Build the custom Docker image based on `docker:dind`
- Start the container with all services
- Create necessary data volumes