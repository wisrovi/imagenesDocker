# Makefile for Nextcloud Docker Deployment

.PHONY: help up down logs certs clean docs-build docs-serve

help:
	@echo "Available commands:"
	@echo "  up           Start the Nextcloud Docker stack (docs available on port 8080)"
	@echo "  down         Stop the Nextcloud Docker stack"
	@echo "  logs         View logs from the Docker stack"
	@echo "  certs        Generate SSL certificates"
	@echo "  clean        Stop stack and remove volumes"
	@echo "  docs-build   Build the Sphinx documentation locally"
	@echo "  docs-serve   Serve the built documentation locally on localhost:8000"

up: certs
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

certs:
	cd openssl && docker-compose up

clean:
	docker-compose down -v

docs-build:
	cd docs && make html

docs-serve: docs-build
	cd docs/_build/html && python -m http.server 8000