# PyPI Traffic Generator - Docker Commands
COMPOSE=docker-compose
IMAGE_NAME=wisrovi/pypi-traffic-generator:latest

.PHONY: help build up down scale logs status clean push

help:
	@echo "PyPI Traffic Generator - Docker Commands"
	@echo "=========================================="
	@echo "  make build      - Construye la imagen de Docker"
	@echo "  make push       - Sube la imagen a Docker Hub"
	@echo "  make up         - Levanta los contenedores (usa REPLICAS en .env)"
	@echo "  make down       - Detiene y elimina los contenedores"
	@echo "  make scale N=x  - Escala a N contenedores (ej: make scale N=5)"
	@echo "  make logs       - Ver logs en tiempo real"
	@echo "  make status     - Ver estado de los contenedores"
	@echo "  make clean      - Borra imágenes, contenedores y volúmenes"

build:
	$(COMPOSE) build

push:
	docker push $(IMAGE_NAME)

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

scale:
	$(COMPOSE) up -d --scale pypi-generator=$(N)

logs:
	$(COMPOSE) logs -f

status:
	$(COMPOSE) ps

clean:
	$(COMPOSE) down --rmi all --volumes --remove-orphans
	docker system prune -f
