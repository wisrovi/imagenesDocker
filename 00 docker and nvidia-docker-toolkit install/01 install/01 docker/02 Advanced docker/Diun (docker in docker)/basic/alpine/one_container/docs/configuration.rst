Configuration
=============

Environment Variables
---------------------

Edit the `.env` file to customize the setup:

- `SSH_PASSWORD`: Password for SSH root access (default: `password`)

  **Security Note**: Change this to a strong password in production environments

Container Configuration
-----------------------

- **Hostname**: The container is configured with hostname `wisrovi` for easy identification in networks

Ports
-----

The following ports are exposed on your host machine:

- `50421`: Portainer web interface
- `50422`: SSH access
- `50423`: Web terminal (ttyd)
- `50424`: HTTP (port 80 inside container)
- `50425`: HTTPS (port 443 inside container)

Volumes
-------

- `./data/dind-data`: Persistent storage for Docker data inside the container
- `./data/portainer_data`: Persistent storage for Portainer configuration and data