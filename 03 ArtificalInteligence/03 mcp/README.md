# MCP Inspector

[![CI](https://github.com/tu-usuario/mcp/workflows/CI/badge.svg)](https://github.com/tu-usuario/mcp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docs.docker.com/compose/)

Este proyecto configura el MCP Inspector usando Docker Compose para facilitar su ejecución.

## Descripción

El MCP Inspector es una herramienta de desarrollo para probar y depurar servidores MCP (Model Context Protocol). Consiste en:

- **Cliente MCP Inspector (MCPI)**: Interfaz web React para interactuar con servidores MCP.
- **Proxy MCP (MCPP)**: Servidor Node.js que actúa como puente entre la UI web y los servidores MCP.

Permite conectar modelos de IA a herramientas, bases de datos y flujos de trabajo externos.

## Instalación y Prerrequisitos

1. Asegúrate de tener Docker y Docker Compose instalados.
2. Ejecuta `make check` para verificar que todo esté configurado correctamente.

## Uso

1. Ejecuta: `make run` o `docker-compose run --rm mcp-inspector`
2. Accede a la interfaz web en `http://localhost:6274` (puerto configurable).

### Comandos disponibles

- `make check` - Verificar prerrequisitos (Docker, Docker Compose)
- `make update` - Actualizar imagen de MCP Inspector
- `make backup` - Crear backup de configuraciones
- `make monitor` - Monitorear estado del contenedor
- `make docs` - Generar documentación local
- `make run` - Ejecutar el inspector (con eliminación automática)
- `make run-fastmcp` - Ejecutar inspector con configuración FastMCP (recomendado)
- `make up` - Ejecutar en modo interactivo (recomendado)
- `make down` - Detener contenedores
- `make logs` - Ver logs en tiempo real
- `make clean` - Limpiar contenedores e imágenes
- `make help` - Mostrar ayuda

## Configuración

### Puertos
- **CLIENT_PORT**: Puerto para la interfaz web (por defecto: 6274)
- **SERVER_PORT**: Puerto para el servidor proxy (por defecto: 6277)

### Timeouts y Opciones Avanzadas
- **MCP_SERVER_REQUEST_TIMEOUT**: Timeout para solicitudes al servidor (ms, por defecto: 300000)
- **MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS**: Resetear timeout en notificaciones de progreso (true/false)
- **MCP_REQUEST_MAX_TOTAL_TIMEOUT**: Timeout máximo total (ms, por defecto: 60000)
- **MCP_AUTO_OPEN_ENABLED**: Abrir navegador automáticamente (true/false)

Modifica el archivo `.env` para cambiar estas configuraciones.

### Archivo de configuración de servidores

Copia y modifica `sample-config.json` para configurar servidores MCP personalizados. El inspector puede cargar configuraciones desde archivos JSON para facilitar el trabajo con múltiples servidores.

Ejemplo de uso con configuración:
```bash
docker-compose run --rm -v $(pwd)/my-config.json:/app/config.json mcp-inspector --config /app/config.json --server my-server
```

## Documentación

La documentación completa del proyecto está disponible en la carpeta `docs/` y puede generarse con Sphinx:

```bash
cd docs
pip install -r requirements.txt
make html
```

Luego abre `docs/_build/html/index.html` en tu navegador.

También puedes ver la documentación generada automáticamente en [GitHub Pages](https://tu-usuario.github.io/mcp/) (si está configurado).

## Contribuir

¡Las contribuciones son bienvenidas! Por favor lee la [guía de contribución](CONTRIBUTING.md) antes de empezar.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md) para el historial de cambios.

## Scripts Adicionales

- `setup.sh` - Configuración inicial automatizada
- `update-image.sh` - Actualiza la imagen de Docker del inspector
- `backup-config.sh` - Crea backups de archivos de configuración
- `check-prerequisites.sh` - Verifica dependencias del sistema
- `monitor.sh` - Monitorea el estado y recursos del contenedor
- `generate-docs.sh` - Genera documentación local con Sphinx

## Desarrollo

Para desarrollo local, puedes usar `docker-compose.override.yml` para personalizar la configuración sin afectar el repositorio principal.

## Plantillas de Issues

El proyecto incluye plantillas para reportar bugs, solicitar funcionalidades y hacer preguntas. Estas ayudan a mantener la calidad de las contribuciones.

## Dockerfile Personalizado

Se incluye un `Dockerfile` opcional para builds personalizados o desarrollo local. Para usarlo:

```bash
docker build -t mcp-inspector-custom .
docker-compose.override.yml  # Configurar para usar imagen local
```

## Acceso Directo al Servidor MCP

Si prefieres acceder directamente a tu servidor MCP sin el proxy del inspector:

```bash
# Inicializar sesión
curl -X POST http://192.168.1.84:8015/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "direct-client", "version": "1.0"}}}'

# Listar herramientas (después de inicializar)
curl -X POST http://192.168.1.84:8015/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'
```

## Soporte

Si necesitas ayuda:

- 📖 **Documentación**: `make docs` o visita la [documentación online](https://tu-usuario.github.io/mcp/)
- 🐛 **Reportar bugs**: Usa las [plantillas de issues](.github/ISSUE_TEMPLATE/)
- 💬 **Preguntas**: Abre un issue con etiqueta `question`
- 🤝 **Contribuciones**: Lee [CONTRIBUTING.md](CONTRIBUTING.md)

## Inicio Rápido

Para usuarios nuevos, ejecuta:

```bash
make setup
make run
```

Esto configurará el proyecto automáticamente y lo iniciará.

## Mejoras realizadas

- Configuración con Docker Compose para facilidad de uso.
- Variables de entorno para puertos configurables.
- Makefile con comandos comunes para facilitar el uso.
- Healthcheck para verificar el estado del servicio.
- Archivo `.gitignore` para ignorar archivos temporales.
- Archivo `sample-config.json` como ejemplo de configuración.
- Scripts de verificación, actualización, backup, monitoreo y docs.
- Archivo `.dockerignore` para optimizar builds.
- `docker-compose.override.yml` para desarrollo local.
- Dockerfile personalizado opcional.
- Documentación completa con Sphinx en carpeta `docs/`.
- CI/CD con GitHub Actions y despliegue automático de docs.
- Guía de contribución y estándares de código.
- Código de conducta y política de seguridad.
- Plantillas de issues para GitHub.
- Archivo de cambios (CHANGELOG).
- Configuración EditorConfig para consistencia.
- Badges en README para estado del proyecto.
- Configuración de funding para soporte al proyecto.
- Script de configuración inicial automatizada.
- Archivo `.env.example` como plantilla de configuración.
- Eliminación de versiones obsoletas en Docker Compose.