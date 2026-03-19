# Ollama Docker Compose Setup with Load Balancing, WebUI, and Secure Tunneling

This project provides a comprehensive Docker Compose configuration for deploying Ollama, an open-source tool for running large language models (LLMs) locally. The setup includes a web-based user interface (Open WebUI), load balancing across multiple Ollama instances for improved performance and redundancy, and secure remote access via Cloudflare Tunnel.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-Instance Ollama Deployment**: Runs two Ollama servers with load balancing for enhanced performance and fault tolerance.
- **Web Interface**: Integrated Open WebUI for easy interaction with AI models through a browser.
- **Load Balancing**: Nginx-based round-robin load balancer distributing requests across Ollama instances.
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support for faster model inference.
- **Secure Remote Access**: Cloudflare Tunnel for secure, authenticated remote access without exposing ports.
- **Data Persistence**: Configurable volumes for storing models and UI data.
- **Health Checks**: Built-in health monitoring for Ollama services.
- **Customizable Environment**: Extensive environment variable configuration for fine-tuning performance.

## Prerequisites

- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **NVIDIA GPU** (optional but recommended for performance):
  - NVIDIA GPU with CUDA support
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Git**: For cloning the repository (optional)

## Project Structure

```
.
├── docker/
│   └── Dockerfile
├── config/
│   └── nginx/
│       └── nginx.conf
├── scripts/
│   └── pull-models.sh
├── tests/
│   └── test.py
├── docs/
│   └── README.md
├── docker-compose.yaml
├── Makefile
└── .gitignore
```

## Architecture

The setup consists of the following services:

### Services Overview

1. **webui** (`ghcr.io/open-webui/open-webui:main`)
   - Provides a web-based interface for interacting with Ollama models
   - Accessible on port 11435 (mapped to internal 8080)
   - Connects to the load balancer for Ollama API calls

2. **ollama_1** and **ollama_2** (`wisrovi/ollama/server:v1.0`)
   - Two identical Ollama server instances
   - Built from the included Dockerfile based on `ollama/ollama`
   - Configured for GPU acceleration with CUDA
   - Expose port 11434 internally
   - Include health checks and optimized environment variables

3. **loadbalancer** (`nginx:latest`)
   - Nginx server configured for load balancing
   - Uses least-connection algorithm for request distribution
   - Routes requests to both Ollama instances
   - Keeps connections alive for better performance

4. **tunnel** (`cloudflare/cloudflared:latest`)
   - Creates a secure tunnel to the web UI
   - Enables remote access without port forwarding
   - Uses Cloudflare's infrastructure for security

### Data Flow

```
User Request → Cloudflare Tunnel → WebUI (Port 8080) → Load Balancer → Ollama Instance (Port 11434)
```

## Installation

1. **Clone the Repository** (optional):
   ```bash
   git clone <repository-url>
   cd ollama-docker-setup
   ```

2. **Ensure Prerequisites**:
   - Install Docker and Docker Compose
   - If using GPU: Install NVIDIA Container Toolkit and verify GPU access

3. **Build Custom Images** (optional):
   ```bash
   make build
   # or
   docker-compose build
   ```

## Configuration

### Environment Variables

Key environment variables for Ollama instances:

- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: 0)
- `OLLAMA_MAX_OFFLOAD`: Maximum GPU memory for offloading (default: 7GB)
- `OLLAMA_FLASH_ATTN`: Enable Flash Attention (1 for enabled, 0 for disabled)
- `OLLAMA_N_BATCH`: Batch size for inference (default: 8)
- `OLLAMA_N_THREADS`: Number of CPU threads (default: 8)
- `NVIDIA_VISIBLE_DEVICES`: GPU visibility (default: all)
- `NVIDIA_DRIVER_CAPABILITIES`: Driver capabilities (default: compute,utility)

### Volumes

- `./ollama_data/ollama_1`: Persistent storage for Ollama instance 1 models
- `./ollama_data/ollama_2`: Persistent storage for Ollama instance 2 models
- `open-webui`: Named volume for WebUI data persistence

### Nginx Configuration

The load balancer is configured in `config/nginx/nginx.conf` with:
- Least-connection load balancing
- Keep-alive connections (32)
- Proxy headers for proper request forwarding

### Cloudflare Tunnel

Currently configured for demo mode. For production use:
1. Create a Cloudflare account
2. Set up a tunnel with authentication
3. Update the tunnel service configuration

## Usage

### Starting the Services

```bash
make start
# or
docker-compose up -d
```

### Starting Individual Services

```bash
# Start only the first Ollama instance
make start-ollama1
# or
docker-compose up -d ollama_1
```

### Accessing the Web Interface

- **Local Access**: http://localhost:11435
- **Remote Access**: Check tunnel logs for Cloudflare URL
  ```bash
  docker-compose logs tunnel
  ```

### Managing Models

Connect to an Ollama container to manage models:

```bash
docker-compose exec ollama_1 bash
ollama pull qwen2.5-coder
ollama list
ollama run qwen2.5-coder
```

### Viewing Logs

```bash
make logs
# or
docker-compose logs -f
```

### Stopping Services

```bash
make stop
# or
docker-compose down
```

### Cleaning Up

Remove containers and volumes:

```bash
make clean
# or
docker-compose down -v
```

## Testing

A Python test script (`tests/test.py`) is included to verify the setup:

```bash
python tests/test.py
```

This script:
- Sends a test prompt to the Ollama API
- Uses the specified model (default: llama3.1:8b)
- Configures context window and temperature
- Prints the generated response

Modify `tests/test.py` variables as needed:
- `model_name`: Change the model to test
- `ollama_host`: Update the host URL if necessary
- `num_ctx`: Adjust context window size
- `temperature`: Set creativity level

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   - Ensure NVIDIA Container Toolkit is installed
   - Check GPU compatibility with `nvidia-smi`
   - Verify Docker runtime: `docker info | grep -i runtime`

2. **Port Conflicts**:
   - Change port mappings in `docker-compose.yaml` if ports are in use
   - Use `netstat -tlnp | grep :11435` to check port usage

3. **Model Download Failures**:
   - Ensure sufficient disk space in volume directories
   - Check network connectivity
   - Verify model names with `ollama list`

4. **Load Balancer Issues**:
   - Check Nginx logs: `docker-compose logs loadbalancer`
   - Verify Ollama instances are healthy: `docker-compose ps`

5. **WebUI Connection Problems**:
   - Confirm load balancer is running
   - Check `OLLAMA_BASE_URL` environment variable
   - Review WebUI logs: `docker-compose logs webui`

### Performance Tuning

- Adjust `OLLAMA_MAX_OFFLOAD` based on GPU memory
- Increase `OLLAMA_N_BATCH` for better throughput (if GPU allows)
- Monitor resource usage with `docker stats`

### Logs and Debugging

- View all logs: `docker-compose logs`
- Follow logs in real-time: `docker-compose logs -f`
- Inspect containers: `docker-compose exec <service> bash`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

**William Rodriguez**  
AI Leader and Solutions Architect at eCaptureDtech  
Location: Badajoz, Extremadura, España  
GitHub: [wisrovi](https://github.com/wisrovi)  
LinkedIn: [wisrovi-rodriguez](https://es.linkedin.com/in/wisrovi-rodriguez)

---

For more information about Ollama, visit [ollama.ai](https://ollama.ai).
For Open WebUI documentation, see [github.com/open-webui/open-webui](https://github.com/open-webui/open-webui).