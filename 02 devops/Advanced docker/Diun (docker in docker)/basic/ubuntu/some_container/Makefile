# Makefile for Docker-in-Docker Basic Setup

.PHONY: docs-build docs-serve docs-clean up

# Define the Docker network name as a constant.
NETWORK_NAME := diun
create_network:
	docker network create $(NETWORK_NAME) || true

# Build the Sphinx documentation
docs-build:
	sphinx-build -b html docs docs/_build/html

# Serve the built documentation
docs-serve: docs-build
	cd docs/_build/html && python3 -m http.server 8000

# Clean the built documentation
docs-clean:
	rm -rf docs/_build

# Default target
help:
	@echo "Available targets:"
	@echo "  docs-build   - Build the Sphinx documentation"
	@echo "  docs-serve   - Build and serve the documentation on localhost:8000"
	@echo "  docs-clean   - Clean the built documentation"
	@echo "  help         - Show this help message"

# Bring up the containers with scale (default SCALE=5)
SCALE ?= 5
up:
	make create_network
	docker-compose up -d dispatcher
	docker-compose up -d --scale worker=$(SCALE)
	sleep 60
	python test/generate_report.py