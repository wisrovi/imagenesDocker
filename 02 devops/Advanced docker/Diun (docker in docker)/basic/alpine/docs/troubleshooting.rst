Troubleshooting
===============

Common Issues
-------------

1. **Port Conflicts**:

   - Ensure ports 50421-50425 are not in use by other services
   - Modify port mappings in `docker-compose.yaml` if needed

2. **Permission Issues**:

   - DinD requires privileged mode, which may be restricted in some environments
   - Ensure Docker daemon on host allows privileged containers

3. **Memory Issues**:

   - DinD can be memory-intensive
   - Increase Docker memory limits if containers fail to start

4. **Portainer Not Accessible**:

   - Check if Portainer container is running: `docker-compose ps`
   - Verify port mapping and firewall settings

5. **SSH Connection Refused**:

   - Ensure SSH service is started inside the container
   - Check SSH password configuration
   - Verify port 50422 is accessible

Logs and Debugging
------------------

::

    # View all service logs
    docker-compose logs

    # View specific service logs
    docker-compose logs dind-basic

    # Access container for manual debugging
    docker-compose exec dind-basic sh

Resetting the Environment
-------------------------

If you encounter persistent issues::

    # Stop and remove everything
    docker-compose down -v --remove-orphans

    # Remove built images
    docker-compose rm -f

    # Clean up unused Docker resources
    docker system prune -f

    # Restart fresh
    docker-compose up -d --build