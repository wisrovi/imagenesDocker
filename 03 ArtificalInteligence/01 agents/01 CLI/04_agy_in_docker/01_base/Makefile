# Makefile for 01_base – Construye y ejecuta la imagen base

.PHONY: all build run clean

all: build run publish
	@echo "✅  01_base – Todo completado"

# Construye la imagen base que contiene el binario agy
build:
	@echo "🔧  Construyendo imagen base (wisrovi/agy-edc:base)…"
	docker build --no-cache --platform linux/amd64 -t wisrovi/agy-edc:base .

# Ejecuta el contenedor base para autenticarse
run:
	@echo "🚀  Lanzando contenedor base para autenticación…"
	./run_base.sh

# Elimina la imagen local (opcional)
tag: build
	@echo "🏷️  Tagging image as 1.0.0..."
	docker tag wisrovi/agy-edc:base wisrovi/agy-edc:1.0.0

publish: tag
	@echo "🚢  Pushing v1.1.0 to registry…"
	docker push wisrovi/agy-edc:v1.0.0
