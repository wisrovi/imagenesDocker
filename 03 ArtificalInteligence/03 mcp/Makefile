# Makefile para MCP Inspector

.PHONY: run up down logs clean check update backup docs-up docs-down help

# Verificar prerrequisitos
check:
	./check-prerequisites.sh

# Actualizar imagen de Docker
update:
	./update-image.sh

# Backup de configuraciones
backup:
	./backup-config.sh

# Ejecutar el inspector (equivalente a docker-compose run --rm)
run: check
	docker-compose run --rm mcp-inspector

# Ejecutar inspector con configuración FastMCP
run-fastmcp: check
	@echo "Ejecutando inspector con configuración FastMCP..."
	docker-compose run --rm -e MCP_CONFIG_FILE=/app/config.json -e MCP_SERVER_NAME=mi-servidor-fastmcp mcp-inspector

# Ejecutar en modo detached
up: check
	docker-compose up -d

# Detener contenedores
down:
	docker-compose down

# Ver logs
logs:
	docker-compose logs -f

# Limpiar contenedores e imágenes no utilizadas
clean:
	docker-compose down --rmi all --volumes --remove-orphans

# Levantar documentación de Sphinx
docs-up: check
	docker-compose up -d sphinx-docs

# Detener documentación de Sphinx
docs-down:
	docker-compose down sphinx-docs

# Mostrar ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  check     - Verificar prerrequisitos (Docker, Docker Compose)"
	@echo "  update    - Actualizar imagen de MCP Inspector"
	@echo "  backup    - Crear backup de configuraciones"
	@echo "  run       - Ejecutar el inspector (con eliminación automática)"
	@echo "  up        - Ejecutar en segundo plano"
	@echo "  down      - Detener contenedores"
	@echo "  logs      - Ver logs en tiempo real"
	@echo "  clean     - Limpiar contenedores e imágenes"
	@echo "  docs-up   - Levantar documentación de Sphinx"
	@echo "  docs-down - Detener documentación de Sphinx"
	@echo "  help      - Mostrar esta ayuda"