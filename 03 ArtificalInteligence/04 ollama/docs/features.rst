Features
========

- **Multi-Instance Ollama Deployment**: Runs two Ollama servers with load balancing for enhanced performance and fault tolerance.
- **Web Interface**: Integrated Open WebUI for easy interaction with AI models through a browser.
- **Load Balancing**: Nginx-based round-robin load balancer distributing requests across Ollama instances.
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support for faster model inference.
- **Secure Remote Access**: Cloudflare Tunnel for secure, authenticated remote access without exposing ports.
- **Data Persistence**: Configurable volumes for storing models and UI data.
- **Health Checks**: Built-in health monitoring for Ollama services.
- **Customizable Environment**: Extensive environment variable configuration for fine-tuning performance.