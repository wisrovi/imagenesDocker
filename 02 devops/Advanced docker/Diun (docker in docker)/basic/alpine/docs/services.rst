Services
========

Docker-in-Docker (DinD)
-----------------------

- Base image: `docker:dind`
- Runs Docker daemon inside the container
- Privileged mode required for DinD functionality

SSH Server
----------

- OpenSSH server running on port 50422
- Root login enabled with configurable password
- Provides secure remote access to the container

Portainer
---------

- Web-based Docker management interface
- Automatically installed and started inside the container
- Accessible at `http://localhost:50421`
- Manages both host and container Docker instances

Web Terminal (ttyd)
-------------------

- Browser-based terminal emulator
- Uses ttyd for web socket connections
- Accessible at `http://localhost:50423`
- Provides full shell access without SSH client