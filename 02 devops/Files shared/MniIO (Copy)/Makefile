# MinIO Docker Setups - Makefile
# This Makefile provides convenient commands for managing the MinIO Docker setups
# and documentation.

.PHONY: help docs-build docs-serve docs-clean minio-normal-up minio-normal-down minio-ssl-up minio-ssl-down minio-ssl-certs clean-all

# Default target
help: ## Show this help message
	@echo "MinIO Docker Setups - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Documentation commands
docs-build: ## Build the Sphinx documentation
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built successfully. Open docs/_build/html/index.html in your browser."

docs-serve: ## Serve the documentation locally (requires docs to be built first)
	@echo "Serving documentation on http://localhost:8000"
	cd docs/_build/html && python3 -m http.server 8000

docs-clean: ## Clean the built documentation
	@echo "Cleaning documentation build files..."
	cd docs && make clean
	rm -rf docs/_build

docs-live: ## Build and serve documentation with live reload (requires sphinx-autobuild)
	@echo "Starting live documentation server..."
	pip install sphinx-autobuild
	sphinx-autobuild docs docs/_build/html --open-browser --port 8000

# MinIO-Normal commands
minio-normal-up: ## Start MinIO-Normal setup
	@echo "Starting MinIO-Normal..."
	cd MinIO-normal && docker-compose up -d
	@echo "MinIO-Normal started. Access console at http://localhost:30707"

minio-normal-down: ## Stop MinIO-Normal setup
	@echo "Stopping MinIO-Normal..."
	cd MinIO-normal && docker-compose down

minio-normal-logs: ## Show MinIO-Normal logs
	@echo "Showing MinIO-Normal logs..."
	cd MinIO-normal && docker-compose logs -f

minio-normal-status: ## Check MinIO-Normal status
	@echo "MinIO-Normal status:"
	cd MinIO-normal && docker-compose ps

# MinIO-SSL commands
minio-ssl-certs: ## Generate SSL certificates for MinIO-SSL
	@echo "Generating SSL certificates..."
	cd Minio-ssl/openssl && docker-compose up
	@echo "Certificates generated in Minio-ssl/certs/"

minio-ssl-up: ## Start MinIO-SSL setup (generates certs if needed)
	@echo "Starting MinIO-SSL..."
	@if [ ! -d "Minio-ssl/certs" ]; then \
		echo "Certificates not found. Generating..."; \
		$(MAKE) minio-ssl-certs; \
	fi
	cd Minio-ssl && docker-compose up -d
	@echo "MinIO-SSL started. Access console at https://localhost:30707"

minio-ssl-down: ## Stop MinIO-SSL setup
	@echo "Stopping MinIO-SSL..."
	cd Minio-ssl && docker-compose down

minio-ssl-logs: ## Show MinIO-SSL logs
	@echo "Showing MinIO-SSL logs..."
	cd Minio-ssl && docker-compose logs -f

minio-ssl-status: ## Check MinIO-SSL status
	@echo "MinIO-SSL status:"
	cd Minio-ssl && docker-compose ps

# Combined commands
minio-up: ## Start both MinIO setups
	@echo "Starting both MinIO setups..."
	$(MAKE) minio-normal-up
	$(MAKE) minio-ssl-up

minio-down: ## Stop both MinIO setups
	@echo "Stopping both MinIO setups..."
	$(MAKE) minio-normal-down
	$(MAKE) minio-ssl-down

# Utility commands
health-check: ## Check health of running MinIO instances
	@echo "Checking MinIO health..."
	@echo "MinIO-Normal health:"
	@curl -s http://localhost:30706/minio/health/live && echo " ✓ Alive" || echo " ✗ Not responding"
	@echo "MinIO-SSL health:"
	@curl -s -k https://localhost:30706/minio/health/live && echo " ✓ Alive" || echo " ✗ Not responding"

install-deps: ## Install required dependencies
	@echo "Installing dependencies..."
	pip install sphinx sphinx-tabs sphinx-copybutton sphinx-rtd-theme boto3 minio dvc
	@echo "Dependencies installed."

setup-dev: ## Set up development environment
	@echo "Setting up development environment..."
	$(MAKE) install-deps
	$(MAKE) docs-build
	@echo "Development environment ready."

clean-all: ## Clean all Docker containers, volumes, and documentation
	@echo "Cleaning all resources..."
	$(MAKE) minio-down
	$(MAKE) docs-clean
	docker system prune -f
	docker volume prune -f
	@echo "All resources cleaned."

# Quick start commands
quick-start-normal: ## Quick start MinIO-Normal with documentation
	@echo "Quick starting MinIO-Normal setup..."
	$(MAKE) minio-normal-up
	$(MAKE) docs-build
	$(MAKE) docs-serve

quick-start-ssl: ## Quick start MinIO-SSL with documentation
	@echo "Quick starting MinIO-SSL setup..."
	$(MAKE) minio-ssl-up
	$(MAKE) docs-build
	$(MAKE) docs-serve

# Testing commands
test-minio-normal: ## Test MinIO-Normal functionality
	@echo "Testing MinIO-Normal..."
	cd MinIO-normal && docker-compose exec dvc-minio mc alias set test http://localhost:9000 DVC uTAntEMTuVpcJucNjOJm
	cd MinIO-normal && docker-compose exec dvc-minio mc mb test/test-bucket
	cd MinIO-normal && docker-compose exec dvc-minio mc cp /dev/null test/test-bucket/test-file
	@echo "MinIO-Normal test completed successfully."

test-minio-ssl: ## Test MinIO-SSL functionality
	@echo "Testing MinIO-SSL..."
	cd Minio-ssl && docker-compose exec dvc-minio mc alias set test https://localhost:9000 DVC uTAntEMTuVpcJucNjOJm --insecure
	cd Minio-ssl && docker-compose exec dvc-minio mc mb test/test-bucket
	cd Minio-ssl && docker-compose exec dvc-minio mc cp /dev/null test/test-bucket/test-file
	@echo "MinIO-SSL test completed successfully."

# Backup commands
backup-normal: ## Create backup of MinIO-Normal data
	@echo "Creating backup of MinIO-Normal data..."
	$(MAKE) minio-normal-down
	tar -czf backup-normal-$(shell date +%Y%m%d-%H%M%S).tar.gz MinIO-normal/DVC_data/
	$(MAKE) minio-normal-up
	@echo "Backup created."

backup-ssl: ## Create backup of MinIO-SSL data
	@echo "Creating backup of MinIO-SSL data..."
	$(MAKE) minio-ssl-down
	tar -czf backup-ssl-$(shell date +%Y%m%d-%H%M%S).tar.gz Minio-ssl/DVC_data/
	$(MAKE) minio-ssl-up
	@echo "Backup created."

# Information commands
info: ## Show project information
	@echo "MinIO Docker Setups"
	@echo "==================="
	@echo "Author: William R. Rodriguez"
	@echo "Repository: MinIO Docker configurations"
	@echo "Documentation: Run 'make docs-serve' to view"
	@echo ""
	@echo "Available setups:"
	@echo "  - MinIO-Normal: Basic setup (http://localhost:30707)"
	@echo "  - MinIO-SSL: SSL-enabled setup (https://localhost:30707)"
	@echo ""
	@echo "Use 'make help' for all available commands."

# Development commands
lint: ## Run linting on Python files (if any)
	@echo "Running linting..."
	# Add linting commands here if Python files are added

format: ## Format code (if any)
	@echo "Formatting code..."
	# Add formatting commands here if code files are added

# CI/CD simulation
ci-build: ## Simulate CI build process
	@echo "Running CI build..."
	$(MAKE) docs-build
	$(MAKE) test-minio-normal
	$(MAKE) test-minio-ssl
	@echo "CI build completed successfully."

# Emergency commands
emergency-stop: ## Emergency stop all MinIO instances
	@echo "Emergency stop - stopping all MinIO instances..."
	docker stop $$(docker ps -q --filter "name=dvc-minio") 2>/dev/null || true
	docker rm $$(docker ps -aq --filter "name=dvc-minio") 2>/dev/null || true
	@echo "Emergency stop completed."