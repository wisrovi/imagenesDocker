Configuration
=============

Environment Variables
---------------------

Key environment variables for Ollama instances:

- ``CUDA_VISIBLE_DEVICES``: GPU device ID (default: 0)
- ``OLLAMA_MAX_OFFLOAD``: Maximum GPU memory for offloading (default: 7GB)
- ``OLLAMA_FLASH_ATTN``: Enable Flash Attention (1 for enabled, 0 for disabled)
- ``OLLAMA_N_BATCH``: Batch size for inference (default: 8)
- ``OLLAMA_N_THREADS``: Number of CPU threads (default: 8)
- ``NVIDIA_VISIBLE_DEVICES``: GPU visibility (default: all)
- ``NVIDIA_DRIVER_CAPABILITIES``: Driver capabilities (default: compute,utility)

Volumes
-------

- ``./ollama_data/ollama_1``: Persistent storage for Ollama instance 1 models
- ``./ollama_data/ollama_2``: Persistent storage for Ollama instance 2 models
- ``open-webui``: Named volume for WebUI data persistence

Nginx Configuration
-------------------

The load balancer is configured in ``config/nginx/nginx.conf`` with:

- Least-connection load balancing
- Keep-alive connections (32)
- Proxy headers for proper request forwarding

Cloudflare Tunnel
-----------------

Currently configured for demo mode. For production use:

1. Create a Cloudflare account
2. Set up a tunnel with authentication
3. Update the tunnel service configuration