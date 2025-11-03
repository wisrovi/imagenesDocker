# Makefile for Docker-in-Docker Basic Setup

.PHONY: docs-build docs-serve docs-clean compose build set-scale up down help


# Define the Docker network name as a constant.
NETWORK_NAME := diun


create_network:
	docker network create $(NETWORK_NAME)

# Build the Sphinx documentation
docs-build:
	sphinx-build -b html docs docs/_build/html

# Serve the built documentation
docs-serve: docs-build
	cd docs/_build/html && python3 -m http.server 8000

# Clean the built documentation
docs-clean:
	rm -rf docs/_build

# Build the Docker images
build:
	docker-compose build

basic_ssh_test:
	sshpass -p 'password' ssh worker1@192.168.1.84 -p 50422

# Set SCALE in docker-compose.yaml
set-scale:
	sed -i 's/SCALE=[0-9]*/SCALE=$(SCALE)/' docker-compose.workers_ssh.yaml

# Bring up the containers with scale (default SCALE=5)
SCALE ?= 5
up: set-scale
	docker-compose -f docker-compose.workers_ssh.yaml up -d --scale worker=$(SCALE) --build

# Bring down the containers
down:
	docker-compose down

# Default target
help:
	@echo "Available targets:"
	@echo "  docs-build     - Build the Sphinx documentation"
	@echo "  docs-serve     - Build and serve the documentation on localhost:8000"
	@echo "  docs-clean     - Clean the built documentation"
	@echo "  compose        - Generate docker-compose.yaml with REPLICAS (e.g., make compose REPLICAS=5)"
	@echo "  build          - Build the Docker images"
	@echo "  set-scale      - Update SCALE in docker-compose.yaml (e.g., make set-scale SCALE=10)"
	@echo "  up             - Bring up the containers with scale (default SCALE=5, e.g., make up SCALE=3)"
	@echo "                  Updates SCALE, rebuilds nginx to update proxy locations. Access services at http://localhost (redirects to HTTPS) or https://localhost"
	@echo "  down           - Bring down the containers"
	@echo "  help           - Show this help message"