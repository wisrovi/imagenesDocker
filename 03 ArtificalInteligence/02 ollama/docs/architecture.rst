Architecture
============

The setup consists of the following services:

Services Overview
-----------------

1. **webui** (``ghcr.io/open-webui/open-webui:main``)

   - Provides a web-based interface for interacting with Ollama models
   - Accessible on port 11435 (mapped to internal 8080)
   - Connects to the load balancer for Ollama API calls

2. **ollama_1** and **ollama_2** (``wisrovi/ollama/server:v1.0``)

   - Two identical Ollama server instances
   - Built from the included Dockerfile based on ``ollama/ollama``
   - Configured for GPU acceleration with CUDA
   - Expose port 11434 internally
   - Include health checks and optimized environment variables

3. **loadbalancer** (``nginx:latest``)

   - Nginx server configured for load balancing
   - Uses least-connection algorithm for request distribution
   - Routes requests to both Ollama instances
   - Keeps connections alive for better performance

4. **tunnel** (``cloudflare/cloudflared:latest``)

   - Creates a secure tunnel to the web UI
   - Enables remote access without port forwarding
   - Uses Cloudflare's infrastructure for security

Data Flow
---------

::

   User Request → Cloudflare Tunnel → WebUI (Port 8080) → Load Balancer → Ollama Instance (Port 11434)