# Makefile for Docker on Docker project

.PHONY: help up down build docs-build docs-serve

help:
	@echo "Available commands:"
	@echo "  up         - Start the Docker container"
	@echo "  down       - Stop the Docker container"
	@echo "  build      - Build the Docker image"
	@echo "  docs-build - Build the Sphinx documentation"
	@echo "  docs-serve - Serve the documentation on localhost:8000"

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

docs-build:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000