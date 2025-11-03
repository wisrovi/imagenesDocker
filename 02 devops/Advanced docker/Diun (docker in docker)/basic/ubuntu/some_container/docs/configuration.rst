Configuration
=============

Environment Variables
---------------------

Create a ``.env`` file in the project root to customize settings::

   SSH_PASSWORD=custom_password

Docker Compose Configuration
----------------------------

The ``docker-compose.yaml`` file defines the service configuration:

- **Privileged Mode**: Required for DinD functionality.
- **Port Mappings**:

  - 50421:9000 - Portainer web interface
  - 50422:50422 - SSH server
  - 50423:7681 - ttyd web terminal
  - 50424:80 - HTTP (if needed)
  - 50425:443 - HTTPS (if needed)
  - 50426:9000 - Portainer agent (alternative port)

- **Volumes**:

  - ``./data/dind-data:/var/lib/docker`` - Persistent Docker data
  - ``./data/portainer_data:/data`` - Portainer data

- **Network**: Custom bridge network for isolation

Dockerfile Details
------------------

The Docker image is based on Ubuntu 22.04 and includes:

- Docker Engine for DinD
- OpenSSH server for remote access
- ttyd for web-based terminal
- tmux and curl for additional utilities
- Non-interactive installation to avoid prompts