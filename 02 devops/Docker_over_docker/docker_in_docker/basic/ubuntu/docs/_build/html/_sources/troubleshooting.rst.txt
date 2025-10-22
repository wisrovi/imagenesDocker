Troubleshooting
===============

Common Issues
-------------

1. **Port Conflicts**: Ensure ports 50421-50426 are available on the host.
2. **Permission Issues**: Run with sudo if Docker requires elevated privileges.
3. **DinD Not Starting**: Check that the container has privileged mode enabled.
4. **SSH Connection Refused**: Verify SSH_PASSWORD is set correctly.

Debugging Commands
------------------

::

   # Check container status
   docker-compose ps

   # View detailed logs
   docker-compose logs dind-basic

   # Restart services
   docker-compose restart

   # Clean up
   docker-compose down -v

Performance Considerations
--------------------------

- DinD can be resource-intensive; monitor host system resources.
- Use appropriate memory limits in production environments.
- Consider using Docker contexts for complex setups.