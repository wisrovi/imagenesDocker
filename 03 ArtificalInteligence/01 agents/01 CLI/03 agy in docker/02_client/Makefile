# Makefile for 02_client – Construye y ejecuta la imagen sin autenticación

.PHONY: all build run clean

all: build run
	@echo "✅  02_client – Todo completado"

# Construye la imagen cliente que copia la configuración .gemini
build:
	@echo "🔧  Copiando backup de autenticación al cliente…"
	mkdir -p agy-config-backup && cp -r ../01_base/agy-config-backup/.gemini agy-config-backup/ || true
	@echo "🔧  Construyendo imagen cliente (wisrovi/agy-edc:1.0.0)…"
	cd $(CURDIR) && docker build -t wisrovi/agy-edc:1.0.0 .

# Ejecuta la imagen cliente (no solicita login)
run:
	@echo "🚀  Lanzando contenedor cliente sin autenticación…"
	docker run -it --rm wisrovi/agy-edc:1.0.0 agy

clean:
	@echo "🧹  Eliminando imagen cliente…"
	-docker rmi -f wisrovi/agy-edc:v1.0.0
