#!/bin/bash

# Mega setup script for Kind cluster with GPU support
# Handles prerequisites, installations, configurations, and testing
# Must be run with sudo: sudo ./scripts/mega_setup.sh

set -o pipefail  # Exit on pipe failures

# Change to project root directory
cd "$(dirname "$0")/.." || { echo "❌ Failed to change to project root"; exit 1; }

# Log file
LOG_FILE="mega_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run with sudo privileges."
    echo "Usage: sudo ./scripts/mega_setup.sh"
    exit 1
fi

# Function to install package if missing
install_if_missing() {
    local package=$1
    if ! dpkg -l | grep -q "^ii  $package "; then
        echo "Installing $package..."
        apt update && apt install -y "$package"
        return $?
    else
        echo "$package already installed"
        return 0
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global success flag
SUCCESS=true
TOTAL_STEPS=9
CURRENT_STEP=0

# Function to print step header
print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "${BLUE}Step $CURRENT_STEP/$TOTAL_STEPS: $1${NC}"
    echo "Remaining steps: $((TOTAL_STEPS - CURRENT_STEP))"
}

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check command success
check_command() {
    if [ $? -eq 0 ]; then
        print_status $GREEN "✅ $1"
    else
        print_status $RED "❌ $1 failed"
        SUCCESS=false
    fi
}

# Initial checks
print_status $YELLOW "Realizando verificaciones iniciales..."

# Check internet connection
if ! curl -s --connect-timeout 5 google.com > /dev/null; then
    print_status $RED "❌ No hay conexión a internet. Requerida para descargas."
    exit 1
fi
print_status $GREEN "✅ Conexión a internet OK"

# Check NVIDIA drivers
if ! nvidia-smi &> /dev/null; then
    print_status $RED "❌ NVIDIA drivers no detectados. Instala los drivers NVIDIA primero."
    exit 1
fi
print_status $GREEN "✅ Drivers NVIDIA OK"

# Install jq if needed for JSON merging
install_if_missing jq
check_command "Instalación de jq (para fusión JSON)"

# Paso 1: Verificar e instalar Docker
# Docker es necesario para ejecutar contenedores, incluyendo los nodos de Kind
print_step "Verificar e instalar Docker"
print_status $YELLOW "Comprobando si Docker está instalado..."
if ! command -v docker &> /dev/null; then
    print_status $YELLOW "Instalando Docker (actualiza repositorios y instala docker.io)..."
    apt update && apt install -y docker.io
    check_command "Instalación de Docker"
    print_status $YELLOW "Iniciando y habilitando el servicio Docker..."
    systemctl start docker
    systemctl enable docker
    check_command "Inicio del servicio Docker"
else
    print_status $GREEN "✅ Docker ya está instalado"
fi

# Paso 2: Verificar e instalar nvidia-container-toolkit
# Este toolkit permite que Docker use el runtime de NVIDIA para contenedores GPU
print_step "Verificar e instalar nvidia-container-toolkit"
print_status $YELLOW "Comprobando si nvidia-container-toolkit está instalado..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    print_status $YELLOW "Instalando nvidia-container-toolkit..."
    apt install -y nvidia-container-toolkit
    check_command "Instalación de nvidia-container-toolkit"
else
    print_status $GREEN "✅ nvidia-container-toolkit ya está instalado"
fi

# Paso 3: Verificar e instalar docker-compose
# Docker Compose facilita la gestión de múltiples contenedores
print_step "Verificar e instalar docker-compose"
print_status $YELLOW "Comprobando si docker-compose está disponible..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_status $YELLOW "Instalando docker-compose-plugin..."
    apt install -y docker-compose-plugin
    check_command "Instalación de docker-compose"
else
    print_status $GREEN "✅ docker-compose ya está disponible"
fi

# Paso 4: Configurar daemon.json de Docker
# Configura el runtime de NVIDIA como predeterminado para usar GPUs en contenedores
print_step "Configurar daemon.json de Docker"
print_status $YELLOW "Configurando /etc/docker/daemon.json para soporte GPU..."
DAEMON_FILE="/etc/docker/daemon.json"
NVIDIA_CONFIG='{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}'

if [ -f "$DAEMON_FILE" ]; then
    print_status $YELLOW "Fusionando configuración con daemon.json existente..."
    # Usa jq para fusionar si está disponible, sino Python
    if command -v jq &> /dev/null; then
        jq '. + '"$NVIDIA_CONFIG" "$DAEMON_FILE" > /tmp/daemon.json && mv /tmp/daemon.json "$DAEMON_FILE"
    else
        python3 -c "
import json
with open('$DAEMON_FILE', 'r') as f:
    existing = json.load(f)
existing.update(json.loads('''$NVIDIA_CONFIG'''))
with open('/tmp/daemon.json', 'w') as f:
    json.dump(existing, f, indent=2)
        " && mv /tmp/daemon.json "$DAEMON_FILE"
    fi
    check_command "Fusión de daemon.json"
else
    print_status $YELLOW "Creando nuevo daemon.json..."
    echo "$NVIDIA_CONFIG" | tee "$DAEMON_FILE" > /dev/null
    check_command "Creación de daemon.json"
fi

# Reiniciar Docker para aplicar cambios
print_status $YELLOW "Reiniciando Docker para aplicar configuración..."
systemctl restart docker
check_command "Reinicio de Docker"

# Paso 5: Instalar Kind y kubectl
# Kind crea clústeres K8s locales; kubectl es el cliente para interactuar con K8s
print_step "Instalar Kind y kubectl"
print_status $YELLOW "Comprobando si Kind está instalado..."
if ! command -v kind &> /dev/null; then
    print_status $YELLOW "Descargando e instalando Kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
    chmod +x ./kind
    mv ./kind /usr/local/bin/kind
    check_command "Instalación de Kind"
else
    print_status $GREEN "✅ Kind ya está instalado"
fi

print_status $YELLOW "Comprobando si kubectl está instalado..."
if ! command -v kubectl &> /dev/null; then
    print_status $YELLOW "Descargando e instalando kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    mv kubectl /usr/local/bin/kubectl
    check_command "Instalación de kubectl"
else
    print_status $GREEN "✅ kubectl ya está instalado"
fi

# Paso 6: Verificar/crear clúster Kind
# Crea el clúster Kubernetes si no existe, usando la configuración con soporte GPU
print_step "Verificar/crear clúster Kind"
print_status $YELLOW "Comprobando si el clúster Kind existe..."
CLUSTER_EXISTS=false
if kind get clusters 2>/dev/null | grep -q "^kind$"; then
    CLUSTER_EXISTS=true
    print_status $GREEN "✅ El clúster Kind ya existe"
    # Verify cluster is healthy before proceeding
    print_status $YELLOW "Verificando salud del clúster existente..."
    if ! kubectl get nodes > /dev/null 2>&1; then
        print_status $YELLOW "Clúster existente no responde, recreando..."
        kind delete cluster
        CLUSTER_EXISTS=false
    fi
fi

if [ "$CLUSTER_EXISTS" = false ]; then
    print_status $YELLOW "Creando clúster Kind con configuración GPU..."
    kind create cluster --config config/kind-config.yaml --wait 300s
    check_command "Creación del clúster"
    # Install NVIDIA device plugin for GPU scheduling
    print_status $YELLOW "Instalando NVIDIA device plugin para scheduling GPU..."
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
    check_command "Instalación del device plugin NVIDIA"
fi

# Paso 7: Asegurar montaje de GPU en workers
# Si el clúster existe, copia nvidia-smi a los nodos worker para pruebas
print_step "Asegurar montaje de GPU en workers"
if [ "$CLUSTER_EXISTS" = true ]; then
    print_status $YELLOW "Copiando nvidia-smi a los nodos worker para pruebas GPU..."
    for worker in kind-worker kind-worker2 kind-worker3 kind-worker4; do
        if docker ps | grep -q "$worker"; then
            docker cp /usr/bin/nvidia-smi "$worker":/usr/bin/nvidia-smi 2>/dev/null || true
        fi
    done
    print_status $GREEN "✅ Montaje de GPU asegurado"
else
    print_status $GREEN "✅ Saltado (clúster nuevo creado con montajes incluidos)"
fi

# Paso 8: Probar funcionalidad del clúster
# Verifica que los nodos estén listos, tengan etiquetas GPU, device plugin y que nvidia-smi funcione
print_step "Probar funcionalidad del clúster"
print_status $YELLOW "Verificando nodos del clúster..."

# Wait for nodes to be ready
print_status $YELLOW "Esperando que los nodos estén Ready..."
NODES_READY=false
for i in {1..30}; do
    if kubectl get nodes --no-headers | awk '{print $2}' | grep -q "Ready"; then
        NODES_READY=true
        break
    fi
    sleep 2
done

if [ "$NODES_READY" = true ]; then
    print_status $GREEN "✅ Verificación de nodos del clúster"
else
    print_status $RED "❌ Verificación de nodos del clúster failed"
    SUCCESS=false
fi

print_status $YELLOW "Verificando etiquetas GPU en nodos..."
GPU_LABELS_OK=true
for node in kind-control-plane kind-worker kind-worker2 kind-worker3 kind-worker4; do
    if kubectl get node "$node" --show-labels 2>/dev/null | grep -q "nvidia.com/gpu.present=true"; then
        print_status $GREEN "✅ Nodo $node tiene etiqueta GPU"
    else
        print_status $RED "❌ Nodo $node no tiene etiqueta GPU"
        GPU_LABELS_OK=false
    fi
done
if [ "$GPU_LABELS_OK" = true ]; then
    print_status $GREEN "✅ Verificación de etiquetas GPU"
else
    print_status $RED "❌ Verificación de etiquetas GPU failed"
    SUCCESS=false
fi

print_status $YELLOW "Verificando NVIDIA device plugin..."
# Wait for device plugin pods to be ready
PLUGIN_READY=false
for i in {1..30}; do
    if kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -q "Running"; then
        PLUGIN_READY=true
        break
    fi
    sleep 2
done

if [ "$PLUGIN_READY" = true ]; then
    print_status $GREEN "✅ Verificación del device plugin NVIDIA"
else
    print_status $RED "❌ Verificación del device plugin NVIDIA failed"
    SUCCESS=false
fi

print_status $YELLOW "Probando acceso GPU en nodos worker..."
GPU_TEST_PASSED=true
for worker in kind-control-plane kind-worker kind-worker2 kind-worker3 kind-worker4; do
    if docker ps --format "table {{.Names}}" | grep -q "^$worker$"; then
        if ! docker exec "$worker" nvidia-smi > /dev/null 2>&1; then
            print_status $RED "❌ Worker $worker no puede acceder a GPU"
            GPU_TEST_PASSED=false
        else
            GPU_INFO=$(docker exec "$worker" nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
            print_status $BLUE "   GPU $worker: $GPU_INFO"
        fi
    else
        print_status $RED "❌ Worker $worker no está corriendo"
        GPU_TEST_PASSED=false
    fi
done
if [ "$GPU_TEST_PASSED" = true ]; then
    print_status $GREEN "✅ Prueba de GPU exitosa"
else
    print_status $RED "❌ Prueba de GPU fallida"
    SUCCESS=false
fi

# Paso 9: Reportar estado final y resumen
# Indica si todo funcionó correctamente o si hubo fallos, y resume lo realizado
print_step "Reportar estado final y resumen"
if [ "$SUCCESS" = true ]; then
    print_status $GREEN "🎉 ¡Todas las verificaciones pasaron! El clúster Kind con soporte GPU está listo."
    echo ""
    echo "📋 Resumen de lo realizado:"
    echo "  - Docker instalado/configurado con runtime NVIDIA"
    echo "  - nvidia-container-toolkit instalado"
    echo "  - docker-compose instalado"
    echo "  - Kind y kubectl instalados"
    echo "  - Clúster Kind creado con 1 control-plane y 4 workers"
    echo "  - GPUs montadas en todos los nodos (nvidia-smi disponible)"
    echo "  - NVIDIA device plugin instalado y corriendo"
    echo "  - Etiquetas GPU configuradas en todos los nodos"
    echo "  - Validación de acceso GPU exitosa en todos los nodos"
    echo ""
    echo "🚀 El clúster está listo para workloads con GPU!"
    echo "   Puedes ejecutar: ./validate_cluster.sh para verificar en cualquier momento"
    echo ""
    echo "Log guardado en: $LOG_FILE"
else
    print_status $RED "❌ Algunos pasos fallaron. Revisa la salida anterior para detalles."
    echo ""
    echo "💡 Para solucionar problemas comunes:"
    echo "   - Verifica que los drivers NVIDIA estén funcionando: nvidia-smi"
    echo "   - Revisa el estado del clúster: kubectl get nodes"
    echo "   - Ejecuta validación completa: ./validate_cluster.sh"
    echo ""
    echo "Log guardado en: $LOG_FILE"
fi