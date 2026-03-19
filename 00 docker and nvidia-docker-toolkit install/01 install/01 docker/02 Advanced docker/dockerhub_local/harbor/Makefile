# Makefile for Harbor Container Registry Setup
# This Makefile provides convenient commands for managing the Harbor installation

.PHONY: help install start stop restart status logs clean upload-images pull-images prepare check-deps docs-build docs-serve

# Default target
help: ## Show this help message
	@echo "Harbor Container Registry Management"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

check-deps: ## Check if Docker and Docker Compose are installed
	@echo "Checking Docker installation..."
	@docker --version || (echo "Docker not found. Please install Docker."; exit 1)
	@echo "Checking Docker Compose installation..."
	@docker-compose --version || docker compose version || (echo "Docker Compose not found. Please install Docker Compose."; exit 1)
	@echo "Dependencies check passed."

prepare: ## Prepare Harbor configuration (run after modifying harbor.yml)
	@echo "Preparing Harbor configuration..."
	./prepare --with-trivy

install: check-deps ## Install and start Harbor (run prepare first if needed)
	@echo "Installing Harbor..."
	./install.sh

start: ## Start Harbor services
	@echo "Starting Harbor services..."
	docker-compose up -d
	@echo "Harbor started. Access at http://$(shell grep '^hostname:' harbor.yml | cut -d' ' -f2):$(shell grep -A1 '^http:' harbor.yml | grep 'port:' | cut -d' ' -f2)"

stop: ## Stop Harbor services
	@echo "Stopping Harbor services..."
	docker-compose down

restart: stop start ## Restart Harbor services

status: ## Show status of Harbor services
	@echo "Harbor services status:"
	docker-compose ps

logs: ## Show logs from all Harbor services
	docker-compose logs -f --tail=100

logs-%: ## Show logs from specific service (e.g., make logs-core)
	docker-compose logs -f --tail=100 $*

clean: stop ## Stop services and remove containers, networks, and volumes
	@echo "Cleaning up Harbor installation..."
	docker-compose down -v
	@echo "Removing data directory..."
	rm -rf data/
	@echo "Cleanup complete. Run 'make install' to reinstall."

upload-images: ## Upload multiple images to Harbor registry
	@echo "Uploading multiple images to Harbor..."
	cd scripts && ./upload_multiple.sh

pull-images: ## Pull images from Harbor registry (example with nginx)
	@echo "Pulling nginx image from Harbor..."
	./scripts/5.sh nginx latest

pull-all-images: ## Pull all available versions of uploaded images
	@echo "Pulling all versions of uploaded images..."
	./scripts/5.sh nginx latest
	./scripts/5.sh alpine latest
	./scripts/5.sh ubuntu latest
	./scripts/5.sh python latest
	./scripts/5.sh python 3.8
	./scripts/5.sh python 3.9
	./scripts/5.sh python 3.10
	./scripts/5.sh python 3.11
	./scripts/5.sh redis latest
	./scripts/5.sh httpd latest
	./scripts/5.sh tomcat latest
	./scripts/5.sh rabbitmq latest
	./scripts/5.sh zookeeper latest

backup: ## Create backup of Harbor data
	@echo "Creating Harbor backup..."
	./backup.sh

restore: ## Restore Harbor from backup
	@echo "Restoring Harbor from backup..."
	./restore.sh

test-connection: ## Test connection to Harbor registry
	@echo "Testing Harbor connection..."
	curl -f http://$(shell grep '^hostname:' harbor.yml | cut -d' ' -f2):$(shell grep -A1 '^http:' harbor.yml | grep 'port:' | cut -d' ' -f2)/api/v2.0/ping || echo "Connection failed"

info: ## Show Harbor information
	@echo "Harbor Information:"
	@echo "Hostname: $(shell grep '^hostname:' harbor.yml | cut -d' ' -f2)"
	@echo "Port: $(shell grep -A1 '^http:' harbor.yml | grep 'port:' | cut -d' ' -f2)"
	@echo "Admin Password: $(shell grep '^harbor_admin_password:' harbor.yml | cut -d' ' -f2)"
	@echo "Data Volume: $(shell grep '^data_volume:' harbor.yml | cut -d' ' -f2)"
	@echo ""
	@echo "Services running:"
	@docker-compose ps --services --filter "status=running"

docs-build: ## Build Sphinx documentation
	@echo "Building Sphinx documentation..."
	cd docs && make html

docs-serve: docs-build ## Serve Sphinx documentation locally
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000