# Makefile for MinIO Docker Compose Setup

.PHONY: help up down logs restart clean docs-build docs-serve

# Default target
help:
	@echo "Available commands:"
	@echo "  make up         - Start MinIO server and docs in detached mode"
	@echo "  make down       - Stop and remove all containers"
	@echo "  make logs       - View MinIO logs"
	@echo "  make restart    - Restart all containers"
	@echo "  make clean      - Stop containers and remove volumes/orphaned containers"
	@echo "  make docs-build - Build Sphinx documentation"
	@echo "  make docs-serve - Serve Sphinx documentation locally on localhost:8000"

# Start MinIO server
up:
	docker-compose up -d

# Stop MinIO server
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Restart containers
restart:
	docker-compose restart

# Clean up: stop containers, remove volumes and orphaned containers
clean:
	docker-compose down --volumes --remove-orphans

# Build Sphinx documentation
docs-build:
	cd docs && make html

# Serve Sphinx documentation
docs-serve:
	cd docs/_build/html && python -m http.server 8000