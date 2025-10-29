# Makefile for MinIO SSL Setup Project

.PHONY: help certs up down restart logs clean docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  certs       - Generate SSL certificates"
	@echo "  up          - Start MinIO server"
	@echo "  down        - Stop MinIO server"
	@echo "  restart     - Restart MinIO server"
	@echo "  logs        - Show MinIO logs"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  docs        - Build Sphinx documentation"
	@echo "  serve-docs  - Serve documentation locally"

# Generate SSL certificates
certs:
	cd openssl && docker-compose up

# Start MinIO server
up:
	docker-compose up -d

# Stop MinIO server
down:
	docker-compose down

# Restart MinIO server
restart: down up

# Show MinIO logs
logs:
	docker-compose logs -f

# Clean up containers and volumes
clean:
	docker-compose down -v
	cd openssl && docker-compose down -v
	docker system prune -f

# Build Sphinx documentation
docs:
	cd docs && make html

# Serve documentation locally
serve-docs: docs
	cd docs/_build/html && python3 -m http.server 8000