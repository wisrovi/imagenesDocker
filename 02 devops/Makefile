.PHONY: help build-docs view-docs clean-docs up down logs

# Default target
help:
	@echo "DevOps Docker Configurations - Main Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  help         Show this help message"
	@echo "  build-docs   Build Sphinx documentation"
	@echo "  view-docs    Open Sphinx documentation in browser"
	@echo "  clean-docs   Clean documentation build files"
	@echo "  up           Start a service (use SERVICE=<directory>)"
	@echo "  down         Stop a service (use SERVICE=<directory>)"
	@echo "  logs         Show logs for a service (use SERVICE=<directory>)"
	@echo ""
	@echo "Examples:"
	@echo "  make build-docs"
	@echo "  make view-docs"
	@echo "  make up SERVICE=balanceo_carga_nginx"
	@echo "  make down SERVICE=balanceo_carga_nginx"
	@echo "  make logs SERVICE=Code\\ QA/SonarQube"

# Documentation commands
build-docs:
	@echo "Building Sphinx documentation..."
	cd docs && make html
	@echo "Documentation built successfully. Use 'make view-docs' to open it."

view-docs:
	@echo "Opening Sphinx documentation in browser..."
	xdg-open docs/_build/html/index.html || open docs/_build/html/index.html || echo "Please open docs/_build/html/index.html manually"

clean-docs:
	@echo "Cleaning documentation build files..."
	cd docs && make clean
	@echo "Documentation build files cleaned."

# Docker Compose commands
up:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: Please specify SERVICE=<service-directory>"; \
		echo "Example: make up SERVICE=balanceo_carga_nginx"; \
		exit 1; \
	fi
	@echo "Starting service: $(SERVICE)"
	cd "$(SERVICE)" && docker-compose up -d
	@echo "Service started. Check logs with: make logs SERVICE=\"$(SERVICE)\""

down:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: Please specify SERVICE=<service-directory>"; \
		echo "Example: make down SERVICE=balanceo_carga_nginx"; \
		exit 1; \
	fi
	@echo "Stopping service: $(SERVICE)"
	cd "$(SERVICE)" && docker-compose down
	@echo "Service stopped."

logs:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: Please specify SERVICE=<service-directory>"; \
		echo "Example: make logs SERVICE=balanceo_carga_nginx"; \
		exit 1; \
	fi
	@echo "Showing logs for service: $(SERVICE)"
	cd "$(SERVICE)" && docker-compose logs -f