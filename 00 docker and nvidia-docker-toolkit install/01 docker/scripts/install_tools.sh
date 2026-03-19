#!/bin/bash

# =============================================================================
#
#  This script automates the installation of various development tools and
#  system utilities on a Linux system. It is designed to set up a
#  comprehensive development environment with a single command.
#
#  The script performs the following actions:
#
#  1.  Installs Snap, a package manager for Linux, and uses it to install:
#      - Notepad-Plus-Plus: A popular source code editor.
#      - Visual Studio Code: A powerful and extensible code editor.
#
#  2.  Installs Google Chrome, a widely used web browser.
#
#  3.  Installs Anydesk, a remote desktop application.
#
#  4.  Installs Curl, a command-line tool for transferring data with URLs.
#
#  5.  Installs Python 3, Pip (the Python package installer), and the
#      following Python libraries:
#      - tqdm: A library for creating progress bars.
#      - selenium: A library for web browser automation.
#
#  6.  Installs system monitoring and cleaning tools:
#      - Stacer: A system optimizer and monitoring tool.
#      - BleachBit: A system cleaner for freeing up disk space.
#      - Timeshift: A system restore utility.
#
#  7.  Installs Lazydocker, a terminal UI for managing Docker containers.
#
#  Usage:
#      - Make the script executable: `chmod +x install_tools.sh`
#      - Run the script: `./install_tools.sh`
#
#  Note:
#      - This script requires 'sudo' privileges to install packages and
#        modify system settings.
#
# =============================================================================

# --- Install Snap and Snap Packages ---
echo "Installing Snap and essential packages..."
sudo apt install -y snap
sudo snap install notepad-plus-plus
sudo snap install --classic code
echo "Snap and packages installed successfully."

# --- Install Google Chrome ---
echo "Installing Google Chrome..."
sudo apt install -y wget
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
rm google-chrome-stable_current_amd64.deb
echo "Google Chrome installed successfully."

# --- Install Anydesk ---
echo "Installing Anydesk..."
sudo wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | sudo apt-key add -
echo "deb http://deb.anydesk.com/ all main" | sudo tee /etc/apt/sources.list.d/anydesk-stable.list
sudo apt update
sudo apt install -y anydesk
echo "Anydesk installed successfully."

# --- Install Curl ---
echo "Installing Curl..."
sudo apt-get install -y curl
echo "Curl installed successfully."

# --- Install Python and Python Packages ---
echo "Installing Python, Pip, and Python packages..."
sudo apt-get install -y python3-pip python3-dev
pip install tqdm selenium
echo "Python and packages installed successfully."

# --- Install System Tools ---
echo "Installing system monitoring and cleaning tools..."
sudo apt install -y stacer bleachbit timeshift
echo "System tools installed successfully."

# --- Install Lazydocker ---
echo "Installing Lazydocker..."
LAZYDOCKER_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazydocker/releases/latest" | grep -Po '"tag_name": "v\K[0-9.]+')
curl -Lo lazydocker.tar.gz "https://github.com/jesseduffield/lazydocker/releases/latest/download/lazydocker_${LAZYDOCKER_VERSION}_Linux_x86_64.tar.gz"
mkdir -p lazydocker-temp
tar xf lazydocker.tar.gz -C lazydocker-temp
sudo mv lazydocker-temp/lazydocker /usr/local/bin
rm -rf lazydocker.tar.gz lazydocker-temp
echo "Lazydocker version ${LAZYDOCKER_VERSION} installed successfully."

echo "All tools have been installed successfully."