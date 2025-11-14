#!/bin/bash

# =============================================================================
#
#  This script automates the setup of Docker containers for a machine learning
#  development environment. It pulls and runs containers for Portainer,
#  NVIDIA CUDA, and TensorFlow with GPU support.
#
#  The script performs the following actions:
#
#  1.  Installs and runs Portainer, a web-based UI for managing Docker
#      containers.
#
#  2.  Tests the NVIDIA GPU integration with Docker by running a container
#      that executes the `nvidia-smi` command.
#
#  3.  Pulls the latest GPU-enabled TensorFlow and TensorFlow-Jupyter
#      Docker images from the official repository.
#
#  4.  Runs a TensorFlow-Jupyter container with GPU support, mounting the
#      current directory as a volume for easy access to notebooks and data.
#
#  5.  Provides a commented-out command for running a development container
#      with a bash shell for interactive development.
#
#  Usage:
#      - Make the script executable: `chmod +x install_containers.sh`
#      - Run the script: `./install_containers.sh`
#
#  Note:
#      - This script requires Docker and the NVIDIA Docker Toolkit to be
#        installed and configured correctly.
#      - It is recommended to run this script from the project's root
#        directory.
#
# =============================================================================

# --- Install and Run Portainer ---
echo "Installing and running Portainer..."
sudo docker run -d -p 8000:8000 -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce
echo "Portainer is running on http://localhost:9000"

# --- Test NVIDIA GPU Integration ---
echo "Testing NVIDIA GPU integration with Docker..."
sudo docker run --gpus all --rm nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
sudo docker run -d --name NVIDIA --gpus all --health-cmd="nvidia-smi || exit 1" --health-interval=30s --health-retries=3 --health-timeout=5s nvidia/cuda:12.2.0-base-ubuntu22.04 bash -c "while true; do nvidia-smi || break; sleep 30; done; tail -f /dev/null"
echo "NVIDIA GPU integration test complete."

# --- Pull TensorFlow Docker Images ---
echo "Pulling TensorFlow Docker images..."
sudo docker pull tensorflow/tensorflow:latest-gpu
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter
echo "TensorFlow Docker images pulled successfully."

# --- Run TensorFlow-Jupyter Container ---
echo "Running TensorFlow-Jupyter container..."
sudo docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v $(pwd):/tf -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-jupyter
echo "TensorFlow-Jupyter container is running."
echo "Access Jupyter at http://localhost:8888"
echo "Access TensorBoard at http://localhost:6006"

# --- Development Container (Optional) ---
# The following command can be used to start a development container with a
# bash shell for interactive development.
#
# echo "To start a development container, run the following command:"
# echo "docker run --gpus all -it --rm --shm-size=16g -v ./:/app -w /app tensorflow/tensorflow:latest-gpu bash"

echo "Container setup complete."
