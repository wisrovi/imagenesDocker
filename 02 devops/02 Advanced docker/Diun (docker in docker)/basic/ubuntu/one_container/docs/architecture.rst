Architecture
============

This setup creates a containerized environment where:

1. The main container runs Ubuntu with Docker installed
2. Docker daemon starts inside the container (DinD)
3. SSH and ttyd provide access methods
4. Portainer runs as a separate container inside the DinD environment
5. Data persistence through mounted volumes

The privileged mode allows the container to access host resources needed for Docker operations.