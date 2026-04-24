FROM alpine:latest

# Instalar dependencias necesarias
RUN apk add --no-cache \
    docker \
    bash \
    curl \
    jq \
    python3

# Configuración del Entorno (Valores por defecto integrados en la imagen)
ENV REPLICAS=5 \
    HOST_IP=127.0.0.1 \
    TARGET_PACKAGE=wpipe \
    LOG_FILE=/app/monitor_pypi.log \
    BATCH_LIMIT=50 \
    SHORT_WAIT_MIN=60 \
    SHORT_WAIT_MAX=300 \
    LONG_WAIT_MIN=1800 \
    LONG_WAIT_MAX=3600 \
    TIMEOUT_SEC=60 \
    PYTHON_VERSIONS="3.8-slim 3.9-slim 3.10-slim 3.11-slim 3.12-slim 3.13-slim" \
    OS_TYPES="linux/x86_64 linux/aarch64 linux/armv7 windows/amd64 windows/arm64 macos/x86_64 macos/arm64 freebsd/amd64" \
    PIP_VERSIONS="20.0 20.1 20.2 20.3 21.0 21.1 21.2 21.3 22.0 22.1 22.2 22.3 23.0 23.1 23.2 23.3 24.0 24.1 24.2 24.3" \
    TG_TOKEN=8657603474:AAGinzBWeU5BgBIMJ2T3BhwAvrAbCQHz80w \
    TG_CHAT_ID=6586101740

WORKDIR /app

# Copiar archivos
COPY app/ /app/
RUN chmod +x /app/run.sh

CMD ["./run.sh"]
