#!/bin/bash

# Script to initialize the pre-commit environment for wisrovi projects
# Usage: chmod +x setup_precommit.sh && ./setup_precommit.sh

echo "--- Initializing pre-commit environment for William Rodríguez (wisrovi) ---"

# 1. Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "[!] pre-commit not found. Installing via pip..."
    pip install pre-commit
else
    echo "[+] pre-commit is already installed."
fi

# 2. Install the hooks defined in .pre-commit-config.yaml
echo "[*] Installing git hooks..."
pre-commit install

# 3. Install the commit-msg hook (required for no-commit-to-branch/conventional commits)
pre-commit install --hook-type commit-msg

# 4. Initialize detect-secrets baseline if it doesn't exist
if [ ! -f .secrets.baseline ]; then
    echo "[*] Initializing detect-secrets baseline..."
    detect-secrets scan > .secrets.baseline
    echo "[+] .secrets.baseline created. Remember to audit it!"
else
    echo "[+] .secrets.baseline already exists."
fi

# 5. Update hooks to the latest versions
echo "[*] Updating hooks to latest versions..."
pre-commit autoupdate

echo "--- Setup complete! Your workflow is now hardened. ---"
