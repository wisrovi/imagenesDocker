#!/bin/bash

# Script: base.sh
# Description: Installs Node Version Manager (nvm) and sets up Node.js version 22 for development environments.
# Usage: ./base.sh
# Prerequisites: curl, bash, zsh or bash shell
# Author: AI Tools Installation Scripts
# Date: 2025-10-08
# Example: Run this script to prepare the environment for Node.js-based tools like npm CLIs.
# Notes: Downloads and installs nvm, sources shell configuration, verifies installation, installs Node.js 22, and sets it as the active version.

# Download and install Node Version Manager (nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Download and install nvm again (redundant, but included as per original)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash




# Source the shell configuration to load nvm into the current session
source ~/.zshrc

# Check the installed nvm version to verify successful installation
nvm --version

# Install Node.js version 22
nvm install 22

# Set Node.js version 22 as the active version for the current shell
nvm use 22