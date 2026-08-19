# AI Agent Environments & Installation Scripts

## Overview

This repository provides a collection of Docker environments, installation scripts, and agent templates for setting up and running various AI-related tools, agents, and development utilities. It is designed to simplify the deployment, configuration, and integration of AI services on both the host system and within containerized development environments.

## Repository Structure

```
.
├── 01 install/            # Host installation scripts for AI tools
│   ├── base.sh            # Installs Node.js v22 via nvm
│   ├── copilot.sh         # Installs GitHub Copilot CLI
│   ├── gemini.sh          # Installs Google Gemini CLI
│   └── opencode.sh        # Installs OpenCode CLI
├── 02 templates/          # Agent skills templates & custom prompt library
│   ├── .agents/skills/    # Predefined skill templates for CLI agents
│   └── prompts/           # Original system prompts for code reviews, testing, etc.
├── 03 docker/             # Custom NVIDIA CUDA container for AI agent development
│   ├── Dockerfile         # CUDA + Zsh + OpenCode + Microsoft Edit build
│   └── docker-compose.yml # Dev environment service configuration
├── 04_agy_in_docker/      # Containerized Google Antigravity (agy) with OAuth persistency
│   ├── 01_base/           # Base image building & initial OAuth login
│   ├── 02_client/         # Pre-authenticated image construction
│   ├── 03_use/            # Simple runtime wrappers
│   └── 04_advance/        # Optimized environment with Zsh, Compose daemon & volume mapping
├── docs/                  # Sphinx documentation source files
├── Makefile               # Project-wide build and task execution commands
└── README.md              # Project root documentation
```

## Technologies & Key Libraries

This project leverages and configures the following key technologies and tools:

* **Docker & Docker Compose** – For isolating and orchestrating AI development environments.
* **NVIDIA CUDA** – GPU acceleration support within containers (using `nvidia/cuda:12.0.0-base-ubuntu22.04`).
* **Google Antigravity (agy)** – An agentic CLI tool for advanced development workflows.
* **OpenCode CLI** – AI-assisted coding command-line utility.
* **GitHub Copilot & Google Gemini CLIs** – Official command-line assistants for interacting with AI models.
* **Zsh & Oh My Zsh** – Customized interactive shell inside containers using the `aussiegeek` theme and helper plugins (`zsh-autosuggestions`, `zsh-syntax-highlighting`, `zsh-completions`).
* **Microsoft Edit** – A simple console-based text editor compiled for Linux.
* **Sphinx & LaTeX** – Used for generating project documentation and exporting technical PDFs/whitepapers.
* **Node.js (v22)** – Required runtime for running Node-based CLI tools.

---

## Directory Details

### 1. Host Installation (`01 install`)
Contains standalone shell scripts to bootstrap AI utilities on your local machine:
* Run `make install` to run all installation scripts sequentially.
* Individual components can be installed using specific targets like `make install-gemini` or `./01\ install/gemini.sh`.

### 2. Agent Skills & Templates (`02 templates`)
A library of modular agent instructions aligning with the `.agents/skills/` standard:
* Features 10 pre-configured skills including automated code reviews, README/Makefile generators, code quality enforcement (Pylint targets), Sphinx/LaTeX documentation workflows, and granular git commit mapping.

### 3. GPU CUDA Agent Container (`03 docker`)
Provides a ready-to-use development environment featuring CUDA integration for GPU workloads:
* Equipped with Zsh, Oh My Zsh plugins, automatic git credentials alignment, `opencode`, and the Microsoft `edit` utility.
* Configurable via `docker-compose.yml` for quick container spin-up.

### 4. Containerized Google Antigravity (`04_agy_in_docker`)
A multi-stage pipeline designed to solve the OAuth authentication persistence challenge when running `agy` inside Docker:
* **Stage 1 (01_base)**: Builds a base image, launches it interactively, and runs the initial OAuth login to store the credentials token in a temporary backup path on the host.
* **Stage 2 (02_client)**: Injects the cached credentials into a new pre-authenticated image.
* **Stage 3 (03_use)**: Runs tasks instantly using the pre-authenticated container.
* **Stage 4 (04_advance)**: An optimized terminal setup with Oh My Zsh, custom aliases, and daemon/Compose mappings.

---

## Quick Start

To quickly build documentation and install essential tools on the host:

```bash
make all
```

To run the custom development environment:

```bash
cd 03\ docker/
docker compose up -d
```

For the Google Antigravity Docker workflow, follow the stage sequence:

```bash
# 1. Authenticate base image
cd 04_agy_in_docker/01_base && make all

# 2. Build pre-authenticated client
cd ../02_client && make all

# 3. Test runtime execution
cd ../03_use && make all

# 4. Spin up advanced Zsh environment
cd ../04_advance && make run
```

## Version History

- **v1.3.0**: Renamed `03 agy in docker` to `04_agy_in_docker` and added `04_advance` stage containing customized Zsh environment and Docker Compose configurations. Updated main README structure and technology matrix.
- **v1.2.0**: Added host installation scripts for Node.js/nvm and Copilot CLI, enhanced docstrings, and expanded inline code explanations.
- **v1.1.0**: Integrated Sphinx documentation pipeline and general-purpose workspace Makefile.
- **v1.0.0**: Initial release featuring host scripts for Gemini CLI, OpenCode, and basic development templates.