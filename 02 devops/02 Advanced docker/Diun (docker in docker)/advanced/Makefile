# Makefile for Docker-in-Docker project

.PHONY: help setup test build up down restart logs clean docs docs-serve stop
.PHONY: tf-init tf-plan tf-apply tf-destroy tf-output backup restore notify

# Default target
help:
	@echo "Available commands:"
	@echo "  setup       - Initial project setup"
	@echo "  test        - Run tests"
	@echo "  build       - Build the Docker images"
	@echo "  up          - Start the services"
	@echo "  down        - Stop the services"
	@echo "  restart     - Restart the services"
	@echo "  logs        - Show logs"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  docs        - Build the Sphinx documentation"
	@echo "  docs-serve  - Build and serve the documentation locally"
	@echo "  stop        - Stop all services"
	@echo ""
	@echo "Terraform commands:"
	@echo "  tf-init     - Initialize Terraform"
	@echo "  tf-plan     - Plan Terraform changes"
	@echo "  tf-apply    - Apply Terraform changes"
	@echo "  tf-destroy  - Destroy Terraform resources"
	@echo "  tf-output   - Show Terraform outputs"
	@echo ""
	@echo "Operations:"
	@echo "  backup      - Create backup of volumes"
	@echo "  restore     - Restore from backup"
	@echo "  notify      - Send test notification"

# Project management
setup:
	./scripts/setup.sh

test:
	./scripts/test.sh

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

stop: down

# Documentation commands
docs:
	cd docs && make html

docs-serve: docs
	cd docs/_build/html && python -m http.server 8000

# Terraform commands
tf-init:
	cd terraform && terraform init

tf-plan: tf-init
	cd terraform && terraform plan

tf-apply: tf-init
	cd terraform && terraform apply -auto-approve

tf-destroy: tf-init
	cd terraform && terraform destroy -auto-approve

tf-output: tf-init
	cd terraform && terraform output

# Operations
backup:
	./scripts/backup.sh

restore:
	@echo "Restore functionality not yet implemented"
	@echo "Please manually extract backup archives to volumes/"

notify:
	./scripts/notify.sh "Test notification from Makefile" "Test Subject" "info"