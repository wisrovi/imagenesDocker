# Makefile for Docker-in-Docker Basic Setup

DOCKER_REGISTRY ?= your-registry.com
IMAGE_NAME_DIND ?= dind-basic
IMAGE_NAME_DOCS ?= dind-docs
TAG ?= latest

.PHONY: help docs docs-build docs-serve docs-clean docs-docker-build docs-docker-push dind-build dind-push

help:
	@echo "Available commands:"
	@echo "  docs-build         Build the Sphinx documentation"
	@echo "  docs-serve         Serve the documentation locally"
	@echo "  docs-clean         Clean the documentation build"
	@echo "  docs-docker-build  Build Docker image for docs"
	@echo "  docs-docker-push   Push docs Docker image"
	@echo "  dind-build         Build Docker image for DinD"
	@echo "  dind-push          Push DinD Docker image"
	@echo "  up                 Start the Docker services"
	@echo "  down               Stop the Docker services"
	@echo "  build              Build the Docker image"

docs-build:
	cd docs && sphinx-build -b html . _build/html

docs-serve: docs-build
	cd docs/_build/html && python3 -m http.server 8000

docs-clean:
	cd docs && rm -rf _build

docs-docker-build: docs-build
	docker build -t $(DOCKER_REGISTRY)/$(IMAGE_NAME_DOCS):$(TAG) docs/

docs-docker-push: docs-docker-build
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME_DOCS):$(TAG)

dind-build:
	docker-compose build

dind-push: dind-build
	docker tag dind-basic $(DOCKER_REGISTRY)/$(IMAGE_NAME_DIND):$(TAG)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME_DIND):$(TAG)

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build