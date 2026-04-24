# 🏠 Local Orchestration Module

Este módulo contiene la lógica de orquestación principal utilizando **Docker Compose**. Permite gestionar múltiples réplicas del generador de tráfico de forma aislada y eficiente.

## 📂 Contenido de la Carpeta

| Archivo | Descripción |
|---------|-------------|
| `app/` | Directorio con el código fuente y activos que se empaquetan en la imagen Docker. |
| `Dockerfile` | Define la construcción de la imagen personalizada basada en Alpine Linux. |
| `docker-compose.yml` | Configura el servicio, volúmenes (Docker Socket) y variables de entorno. |
| `Makefile` | Atajos de comandos para facilitar la gestión (build, up, scale, logs). |
| `.env` | Archivo de configuración crítica (IP, Paquete, Tokens de Telegram). |

## 🚀 Guía de Uso Rápido

1. **Configuración**: Edita el archivo `.env` con tus credenciales y el paquete objetivo.
2. **Construcción**: Ejecuta `make build` para empaquetar el script y las imágenes base.
3. **Ejecución**:
   - `make up`: Inicia el sistema con las réplicas definidas en `.env`.
   - `make scale N=10`: Escala dinámicamente a 10 instancias.
4. **Monitoreo**: `make logs` para ver la actividad de todas las réplicas en tiempo real.

## 🛠️ Detalles Técnicos
El sistema utiliza **DooD (Docker-out-of-Docker)** montando `/var/run/docker.sock`. Esto permite que el contenedor Alpine gestione contenedores de Python directamente en el host, optimizando el rendimiento y el consumo de recursos.
