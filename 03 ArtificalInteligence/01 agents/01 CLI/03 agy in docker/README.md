# agy‑edc project

## Overview
This repository contains a small Docker‑based workflow to use **Google Antigravity (agy)** without needing to re‑authenticate on every container run.

The project is organized into three folders:

- `01_base` – Builds a base image that installs the `agy` binary and is used to perform the initial OAuth login.
- `02_client` – Builds a second image that copies the authentication configuration (`$HOME/.gemini`) from the host, so `agy` starts without prompting for login.
- `03_use` – Provides a simple script to run the final image (`wisrovi/agy-edc:v1.1.0`).

Each folder ships its own **Makefile** with `all`, `build`, `run` (and `clean`) targets, so you can simply execute `make all` inside the folder.

## Technologies & Key Libraries
- **Base image (`wisrovi/agy-edc:base`) → tagged as `v1.1.0` for production**
- **Ubuntu 24.04** – base image.
- **agy** – Google Antigravity CLI (downloaded directly from the official releases).
- **Make** – orchestrates builds and runs.
- **Bash** – scripts for building/running containers.

## Quick start
```bash
# 1️⃣ Build and authenticate with the base image
cd 01_base
make all   # builds the image and opens an interactive agy session to login

# 2️⃣ Build the client image that re‑uses the stored token
cd ../02_client
make all   # copies the .gemini config and runs the image (no login prompt)

# 3️⃣ Run the final image
cd ../03_use
make all   # runs the published image
```

## Development notes
- The Dockerfile in `01_base` uses `ARG TARGETARCH` to automatically download the correct binary for the host architecture.
- The `Makefile` in `01_base` forces an `amd64` build (`--platform linux/amd64`) with `--no-cache` to avoid using stale layers.
- All changes are tracked with one‑file‑per‑commit Git history.

---
*Author: WILLIAM R.*, AI Leader & Solutions Architect at eCaptureDtech
