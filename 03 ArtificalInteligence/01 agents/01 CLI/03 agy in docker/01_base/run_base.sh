#!/usr/bin/env bash
# run_base.sh – builds the base image, runs agy for authentication,
# and copies the generated config into agy-config-backup/.gemini

set -euo pipefail

# Directory of this script (01_base)
cd "$(dirname "$0")"

# -----------------------------------------------------------------
# 1. Build the base image (contains agy binary)
# -----------------------------------------------
docker build --no-cache --platform linux/amd64 -t wisrovi/agy-edc:base .

# -----------------------------------------------------------------
# 2. Run the container mounting the host config directory where agy stores its token
# -----------------------------------------------------------------
# Run as the current host user so files created inside the container are owned by you
USER_ID=$(id -u)
GROUP_ID=$(id -g)

docker run -it --rm \
  -v "${HOME}/.config/agy-edc:/root/.config/agy-edc:rw" \
  -v "${HOME}/.gemini:/root/.gemini:rw" \
  wisrovi/agy-edc:base agy

# -----------------------------------------------------------------
# 3. After the user exits the container (type `exit`), copy the auth config to the backup folder
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# 3. After the user exits the container (type `exit`), copy the auth config to the backup folder
# -----------------------------------------------------------------
BACKUP_DIR="agy-config-backup/.gemini"
# Remove any previous backup to avoid permission‑conflict files
rm -rf "${BACKUP_DIR}"/*
mkdir -p "${BACKUP_DIR}"
# Copy the *contents* of the .gemini directory (preserving permissions)
cp -r "${HOME}/.config/agy-edc/.gemini/." "${BACKUP_DIR}/"

