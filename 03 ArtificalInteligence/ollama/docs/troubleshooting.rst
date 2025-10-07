Troubleshooting
===============

Common Issues
-------------

1. **GPU Not Detected**:

   - Ensure NVIDIA Container Toolkit is installed
   - Check GPU compatibility with ``nvidia-smi``
   - Verify Docker runtime: ``docker info | grep -i runtime``

2. **Port Conflicts**:

   - Change port mappings in ``docker-compose.yaml`` if ports are in use
   - Use ``netstat -tlnp | grep :11435`` to check port usage

3. **Model Download Failures**:

   - Ensure sufficient disk space in volume directories
   - Check network connectivity
   - Verify model names with ``ollama list``

4. **Load Balancer Issues**:

   - Check Nginx logs: ``docker-compose logs loadbalancer``
   - Verify Ollama instances are healthy: ``docker-compose ps``

5. **WebUI Connection Problems**:

   - Confirm load balancer is running
   - Check ``OLLAMA_BASE_URL`` environment variable
   - Review WebUI logs: ``docker-compose logs webui``

Performance Tuning
------------------

- Adjust ``OLLAMA_MAX_OFFLOAD`` based on GPU memory
- Increase ``OLLAMA_N_BATCH`` for better throughput (if GPU allows)
- Monitor resource usage with ``docker stats``

Logs and Debugging
------------------

- View all logs: ``docker-compose logs``
- Follow logs in real-time: ``docker-compose logs -f``
- Inspect containers: ``docker-compose exec <service> bash``