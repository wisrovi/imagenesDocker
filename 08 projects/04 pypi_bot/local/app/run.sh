#!/bin/bash

# ==========================================
# CONFIGURACIÓN GLOBAL (con soporte para variables de entorno)
# ==========================================

# Si las variables de entorno no están definidas, usar valores por defecto
TARGET_PACKAGE="${TARGET_PACKAGE:-wpipe}"
BATCH_LIMIT="${BATCH_LIMIT:-50}"
SHORT_WAIT_MIN="${SHORT_WAIT_MIN:-60}"
SHORT_WAIT_MAX="${SHORT_WAIT_MAX:-300}"
LONG_WAIT_MIN="${LONG_WAIT_MIN:-1800}"
LONG_WAIT_MAX="${LONG_WAIT_MAX:-3600}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"

# Telegram Config (desde variables de entorno)
TG_TOKEN="${TG_TOKEN:-}"
TG_CHAT_ID="${TG_CHAT_ID:-}"

# Recursos
LOG_FILE="${LOG_FILE:-monitor_pypi.log}"
IMAGES_TAR="${IMAGES_TAR:-docker_images.tar}"

# Versiones de Python (parsear desde variable de entorno o usar default)
if [ -n "$PYTHON_VERSIONS" ]; then
    read -ra VERSIONS <<< "$PYTHON_VERSIONS"
else
    VERSIONS=("3.8-slim" "3.9-slim" "3.10-slim" "3.11-slim" "3.12-slim" "3.13-slim")
fi

# Tipos de OS
if [ -n "$OS_TYPES" ]; then
    read -ra OS_TYPES <<< "$OS_TYPES"
else
    OS_TYPES=("linux/x86_64" "linux/aarch64" "linux/armv7" "windows/amd64" "windows/arm64" "macos/x86_64" "macos/arm64" "freebsd/amd64")
fi

# Versiones de pip
if [ -n "$PIP_VERSIONS" ]; then
    read -ra PIP_VERSIONS <<< "$PIP_VERSIONS"
else
    PIP_VERSIONS=("20.0" "20.1" "20.2" "20.3" "21.0" "21.1" "21.2" "21.3" "22.0" "22.1" "22.2" "22.3" "23.0" "23.1" "23.2" "23.3" "24.0" "24.1" "24.2" "24.3")
fi

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Obtener IP local del equipo (priorizar la del host si se pasa por env)
DISPLAY_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"

send_telegram() {
    local message=$1
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
            -d chat_id="$TG_CHAT_ID" \
            -d text="[${DISPLAY_IP}] $message" > /dev/null
    fi
}

countdown() {
    local secs=$1
    while [ $secs -gt 0 ]; do
        echo -ne "  ${CYAN}[Wait]${NC} Siguiente ciclo en: ${YELLOW}$secs${NC} segundos...\r"
        sleep 1
        : $((secs--))
    done
    echo -e "\n"
}

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}[ERROR]${NC} Docker no está corriendo. Inícialo primero."
        exit 1
    fi
}

get_package_versions() {
    curl -s "https://pypi.org/pypi/$TARGET_PACKAGE/json" 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(' '.join(d['releases'].keys()))" 2>/dev/null || echo ""
}

load_or_pull_images() {
    # Verificar si al menos una de las imágenes necesarias ya existe en el host
    local first_img="python:${VERSIONS[0]}"
    if docker image inspect "$first_img" >/dev/null 2>&1; then
        echo -e "${GREEN}[IMAGES]${NC} Las imágenes de Python ya están presentes en el host. Saltando carga."
        return 0
    fi

    if [ -f "$IMAGES_TAR" ]; then
        echo -e "${CYAN}[IMAGES]${NC} Cargando imágenes desde $IMAGES_TAR (esto puede tardar la primera vez)..."
        docker load -i "$IMAGES_TAR"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[IMAGES]${NC} Imágenes cargadas exitosamente."
            return 0
        else
            echo -e "${YELLOW}[IMAGES]${NC} Error al cargar el .tar. Intentando descargar..."
            pull_and_save_images
        fi
    else
        echo -e "${YELLOW}[IMAGES]${NC} No se encontró $IMAGES_TAR dentro del contenedor. Descargando..."
        pull_and_save_images
    fi
}

pull_and_save_images() {
    echo -e "${CYAN}[IMAGES]${NC} Descargando imágenes de Python..."
    for img in "${VERSIONS[@]}"; do
        echo -e "  Descargando python:$img..."
        docker pull python:$img
    done
    echo -e "${CYAN}[IMAGES]${NC} Guardando en $IMAGES_TAR..."
    docker save -o "$IMAGES_TAR" "${VERSIONS[@]/#/python:}"
    echo -e "${GREEN}[IMAGES]${NC} Imágenes guardadas."
}

echo -e "${CYAN}[PKG]${NC} Obteniendo versiones disponibles de $TARGET_PACKAGE..."
PKG_VERSIONS=($(get_package_versions))
if [ ${#PKG_VERSIONS[@]} -eq 0 ] || [ -z "${PKG_VERSIONS[0]}" ]; then
    echo -e "${YELLOW}[PKG]${NC} No se pudieron obtener versiones. Usando instalación sin versión específica."
    PKG_VERSIONS=("")
fi
echo -e "${GREEN}[PKG]${NC} Versiones encontradas: ${#PKG_VERSIONS[@]}"

# ==========================================
# INICIALIZACIÓN
# ==========================================
check_docker
load_or_pull_images
SESSION_SUCCESS=0
TOTAL_CUMULATIVE=0

clear
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}   PYPI TRAFFIC GENERATOR - $TARGET_PACKAGE ${NC}"
echo -e "${GREEN}==========================================${NC}"
send_telegram "🚀 Sistema iniciado para el paquete: $TARGET_PACKAGE"

trap "echo -e '\n${RED}[!] Deteniendo...${NC}'; send_telegram '🛑 Script apagado. Total: $TOTAL_CUMULATIVE'; exit" SIGINT

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
while true
do
    # Aleatoriedad
    V_PY=${VERSIONS[$RANDOM % ${#VERSIONS[@]}]}
    V_OS=${OS_TYPES[$RANDOM % ${#OS_TYPES[@]}]}
    V_PIP=${PIP_VERSIONS[$RANDOM % ${#PIP_VERSIONS[@]}]}
    
    # Seleccionar solo latest (última versión)
    V_PKG_VER=""
    
# Tipo de operación: pip install, pip download, pipenv, poetry (con weights)
OP_TYPE=$((RANDOM % 10))
if [ $OP_TYPE -lt 5 ]; then
    OP_TYPE=0  # 50% pip install
elif [ $OP_TYPE -lt 8 ]; then
    OP_TYPE=1  # 30% pip download
elif [ $OP_TYPE -lt 9 ]; then
    OP_TYPE=2  # 10% pipenv
else
    OP_TYPE=3  # 10% poetry
fi
    
    # Variar user-agent con más detalles
    OS_DETAIL="Ubuntu 22.04"
    ARCH="x86_64"
    case $V_OS in
        "linux/x86_64") OS_DETAIL="Ubuntu 22.04"; ARCH="x86_64" ;;
        "linux/aarch64") OS_DETAIL="Ubuntu 22.04"; ARCH="aarch64" ;;
        "linux/armv7") OS_DETAIL="Ubuntu 22.04"; ARCH="armv7l" ;;
        "windows/amd64") OS_DETAIL="Windows 11"; ARCH="amd64" ;;
        "windows/arm64") OS_DETAIL="Windows 11"; ARCH="arm64" ;;
        "macos/x86_64") OS_DETAIL="macOS 14"; ARCH="x86_64" ;;
        "macos/arm64") OS_DETAIL="macOS 14"; ARCH="arm64" ;;
        "freebsd/amd64") OS_DETAIL="FreeBSD 13"; ARCH="amd64" ;;
    esac
    
    # Ejecutar operación
    echo -e "${CYAN}[$(date +%T)]${NC} Preparando entorno: ${YELLOW}Python $V_PY | $V_OS | $V_PIP | op:$OP_TYPE${NC}"
    
    case $OP_TYPE in
        0) # pip install
            CMD="pip install$V_PKG_VER $TARGET_PACKAGE 2>&1 | head -5"
            ;;
        1) # pip download
            CMD="pip download$V_PKG_VER $TARGET_PACKAGE 2>&1 | head -5"
            ;;
        2) # pipenv install
            CMD="pip install pipenv 2>&1 | tail -1 && cd /tmp && pipenv install $TARGET_PACKAGE 2>&1 | tail -5"
            ;;
        3) # poetry add
            CMD="pip install poetry 2>&1 | tail -1 && cd /tmp && poetry new temp_proj 2>&1 >/dev/null && cd temp_proj && poetry add $TARGET_PACKAGE 2>&1 | tail -5"
            ;;
    esac
    
    # Agregar "ruido" - headers adicionales para parecer más humano
    ACCEPT_ENCODING="gzip, deflate"
    ACCEPT_LANG="en-US,en;q=0.9,es;q=0.8"
    RANDOM_EXTRA=$(printf "%.0s" $(seq 1 $((1 + RANDOM % 3))))
    
    timeout $TIMEOUT_SEC docker run --rm \
      --env-file /dev/null \
      -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
      -e PIP_USER_AGENT="pip/$V_PIP (python/$V_PY; $OS_DETAIL; $ARCH)($V_OS) $RANDOM_EXTRA" \
      -e PIP_INDEX_URL="https://pypi.org/simple" \
      python:$V_PY bash -c "export HTTP_TIMEOUT=$TIMEOUT_SEC && $CMD"

    if [ $? -eq 0 ]; then
        ((SESSION_SUCCESS++))
        ((TOTAL_CUMULATIVE++))
        echo -e "  ${GREEN}✓${NC} Instalación exitosa (#$TOTAL_CUMULATIVE)"
        echo "[$(date +%T)] OK - $V_PY - $V_OS - pkg$V_PKG_VER" >> $LOG_FILE
    else
        echo -e "  ${RED}✗${NC} Error en la descarga."
        echo "[$(date +%T)] FAIL - $V_PY" >> $LOG_FILE
    fi

    # No borramos imágenes para mantenerlas cacheadas (más rápido)
    # docker rmi python:$V_PY > /dev/null 2>&1
    # docker system prune -f > /dev/null 2>&1

    # Lógica de bloques
    if [ $SESSION_SUCCESS -ge $BATCH_LIMIT ]; then
        L_WAIT=$((LONG_WAIT_MIN + RANDOM % (LONG_WAIT_MAX - LONG_WAIT_MIN + 1)))
        MSG="✅ Bloque completado ($BATCH_LIMIT). Total: $TOTAL_CUMULATIVE. Descansando $((L_WAIT/60)) min."
        echo -e "${YELLOW}$MSG${NC}"
        send_telegram "$MSG"
        SESSION_SUCCESS=0
        countdown $L_WAIT
    else
        S_WAIT=$((SHORT_WAIT_MIN + RANDOM % (SHORT_WAIT_MAX - SHORT_WAIT_MIN + 1)))
        countdown $S_WAIT
    fi
done
