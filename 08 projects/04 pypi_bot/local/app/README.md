# 📦 Core Application Logic

Este directorio contiene los archivos que se copian **dentro** de la imagen personalizada de Docker. Es el "corazón" operativo del sistema.

## 📂 Contenido del Módulo

| Archivo | Propósito |
|---------|-----------|
| `run.sh` | Script principal en Bash que orquestra las descargas, varía User-Agents y maneja la comunicación con Telegram. |
| `docker_images.tar` | Un archivo comprimido con las imágenes oficiales de Python (v3.8-3.13) para evitar descargas innecesarias desde Docker Hub. |

## ⚙️ Funcionamiento Interno

Cuando el contenedor arranca, ejecuta el script `run.sh` el cual:
1. **Verificación de Docker**: Confirma que tiene acceso al socket del host.
2. **Carga Inteligente**: Solo carga las imágenes del archivo `.tar` si no están ya presentes en el host, optimizando el inicio de múltiples réplicas.
3. **Ciclo de Simulación**: Genera entornos de Python aleatorios, configura User-Agents que imitan diversos sistemas operativos y realiza la operación solicitada (install, download, etc.).
4. **Notificación**: Envía el progreso a Telegram identificando la máquina mediante la `HOST_IP` configurada en el entorno.
