#!/bin/bash

# =============================================================================
#
#  This script automates the installation and configuration of Docker,
#  Docker Compose, and the NVIDIA Docker Toolkit on a Linux system. It is
#  designed to streamline the setup process for a development environment
#  that requires GPU acceleration within Docker containers.
#
#  The script performs the following actions:
#
#  1.  NVIDIA Driver Installation:
#      - Detects the appropriate NVIDIA drivers for the system's hardware.
#      - Automatically installs the recommended drivers.
#      - Reboots the system to apply the driver installation.
#
#  2.  Docker Installation:
#      - Downloads and executes the official Docker installation script.
#      - Adds the current user to the 'docker' group to allow running Docker
#        commands without 'sudo'.
#
#  3.  Docker Compose Installation:
#      - Fetches the latest version of Docker Compose from the official
#        GitHub repository.
#      - Downloads the appropriate binary for the system's architecture.
#      - Makes the Docker Compose binary executable and moves it to a
#        directory in the system's PATH.
#
#  4.  NVIDIA Docker Toolkit Installation:
#      - Adds the NVIDIA Docker repository to the system's package manager.
#      - Installs the NVIDIA Container Toolkit, which enables GPU support
#        in Docker containers.
#      - Restarts the Docker service to apply the changes.
#
#  5.  System Cleanup:
#      - Removes any unused packages to free up disk space.
#
#  Usage:
#      - Make the script executable: `chmod +x install_docker.sh`
#      - Run the script: `./install_docker.sh`
#
#  Note:
#      - This script requires 'sudo' privileges to install packages and
#        modify system settings.
#      - A system reboot is required after the NVIDIA drivers are installed.
#
# =============================================================================

# --- NVIDIA Driver Installation ---
# For more information, see:
# https://www.datamachinist.com/deep-learning/install-tensorflow-2-0-using-docker-with-gpu-support-on-ubuntu-18-04/
# https://la.nvidia.com/Download/driverResults.aspx/193108/la

echo "Detecting and installing NVIDIA drivers..."
sudo apt-get install ubuntu-drivers-common 
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
echo "NVIDIA drivers installed. A system reboot is required."
sudo reboot

# --- Docker Installation ---
echo "Installing Docker..."
curl -sSL https://get.docker.com | sh
sudo usermod -a -G docker $USER
echo "Docker installed successfully."

# --- Docker Compose Installation ---
echo "Installing Docker Compose..."
VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*\d')
DESTINATION=/usr/local/bin/docker-compose
sudo curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-$(uname -s)-$(uname -m) -o $DESTINATION
sudo chmod 755 $DESTINATION
echo "Docker Compose version ${VERSION} installed successfully."

# --- NVIDIA Docker Toolkit Installation ---
echo "Installing NVIDIA Docker Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "NVIDIA Docker Toolkit installed successfully."

# --- System Cleanup ---
echo "Removing unused packages..."
sudo apt autoremove -y
echo "System cleanup complete."

echo "Installation complete. Please log out and log back in for the changes to take effect."