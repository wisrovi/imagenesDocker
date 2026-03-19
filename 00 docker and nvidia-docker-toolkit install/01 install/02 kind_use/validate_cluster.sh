#!/bin/bash

# Validation script for Kind cluster with GPU support
# Checks that all workers exist and can access GPU

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global success flag
SUCCESS=true

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

echo -e "${BLUE}=== Validación de Clúster Kind con GPU ===${NC}"
echo ""

# Check if Kind cluster exists
print_status $YELLOW "Verificando si el clúster Kind existe..."
if ! kind get clusters 2>/dev/null | grep -q "^kind$"; then
    print_status $RED "❌ El clúster Kind no existe"
    exit 1
fi
print_status $GREEN "✅ Clúster Kind encontrado"

# Check kubectl connection
print_status $YELLOW "Verificando conexión con kubectl..."
if ! kubectl cluster-info --context kind-kind > /dev/null 2>&1; then
    print_status $RED "❌ No se puede conectar al clúster con kubectl"
    exit 1
fi
print_status $GREEN "✅ Conexión kubectl exitosa"

# Get all expected nodes
EXPECTED_NODES=("kind-control-plane" "kind-worker" "kind-worker2" "kind-worker3" "kind-worker4")

print_status $YELLOW "Verificando que todos los nodos esperados existen..."
ALL_NODES_EXIST=true
for node in "${EXPECTED_NODES[@]}"; do
    if kubectl get nodes | grep -q "$node"; then
        print_status $GREEN "✅ Nodo $node encontrado"
    else
        print_status $RED "❌ Nodo $node no encontrado"
        ALL_NODES_EXIST=false
    fi
done

if [ "$ALL_NODES_EXIST" = true ]; then
    print_status $GREEN "✅ Todos los nodos esperados existen"
else
    print_status $RED "❌ Faltan algunos nodos esperados"
    SUCCESS=false
fi

# Check node status
print_status $YELLOW "Verificando estado de los nodos..."
ALL_NODES_READY=true
for node in "${EXPECTED_NODES[@]}"; do
    if kubectl get nodes | grep "$node" | grep -q "Ready"; then
        print_status $GREEN "✅ Nodo $node está Ready"
    else
        print_status $RED "❌ Nodo $node no está Ready"
        ALL_NODES_READY=false
    fi
done

if [ "$ALL_NODES_READY" = true ]; then
    print_status $GREEN "✅ Todos los nodos están Ready"
else
    print_status $RED "❌ Algunos nodos no están Ready"
    SUCCESS=false
fi

# Check GPU access in control-plane
print_status $YELLOW "Verificando acceso GPU en control-plane..."
if docker exec kind-control-plane nvidia-smi > /dev/null 2>&1; then
    print_status $GREEN "✅ Control-plane puede acceder a GPU"
    GPU_INFO_CP=$(docker exec kind-control-plane nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    print_status $BLUE "   GPU: $GPU_INFO_CP"
else
    print_status $RED "❌ Control-plane no puede acceder a GPU"
    SUCCESS=false
fi

# Check GPU access in worker nodes
print_status $YELLOW "Verificando acceso GPU en nodos worker..."
WORKER_NODES=("kind-worker" "kind-worker2" "kind-worker3" "kind-worker4")
ALL_WORKERS_GPU=true

for worker in "${WORKER_NODES[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "^$worker$"; then
        if docker exec "$worker" nvidia-smi > /dev/null 2>&1; then
            print_status $GREEN "✅ Worker $worker puede acceder a GPU"
            GPU_INFO=$(docker exec "$worker" nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
            print_status $BLUE "   GPU: $GPU_INFO"
        else
            print_status $RED "❌ Worker $worker no puede acceder a GPU"
            ALL_WORKERS_GPU=false
        fi
    else
        print_status $RED "❌ Worker $worker no está corriendo"
        ALL_WORKERS_GPU=false
    fi
done

if [ "$ALL_WORKERS_GPU" = true ]; then
    print_status $GREEN "✅ Todos los workers pueden acceder a GPU"
else
    print_status $RED "❌ Algunos workers no pueden acceder a GPU"
    SUCCESS=false
fi

# Check NVIDIA device plugin
print_status $YELLOW "Verificando NVIDIA device plugin..."
if kubectl get pods -n kube-system | grep -q nvidia-device-plugin; then
    print_status $GREEN "✅ NVIDIA device plugin encontrado"
    PLUGIN_PODS=$(kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds -o jsonpath='{.items[*].status.phase}' 2>/dev/null)
    if [[ "$PLUGIN_PODS" == *"Running"* ]]; then
        print_status $GREEN "✅ Device plugin pods están Running"
    else
        print_status $RED "❌ Device plugin pods no están Running"
        SUCCESS=false
    fi
else
    print_status $RED "❌ NVIDIA device plugin no encontrado"
    SUCCESS=false
fi

# Check GPU labels
print_status $YELLOW "Verificando etiquetas GPU en nodos..."
GPU_LABELS_OK=true
for node in "${EXPECTED_NODES[@]}"; do
    if kubectl get node "$node" --show-labels | grep -q "nvidia.com/gpu.present=true"; then
        print_status $GREEN "✅ Nodo $node tiene etiqueta GPU"
    else
        print_status $RED "❌ Nodo $node no tiene etiqueta GPU"
        GPU_LABELS_OK=false
    fi
done

if [ "$GPU_LABELS_OK" = true ]; then
    print_status $GREEN "✅ Todos los nodos tienen etiquetas GPU"
else
    print_status $RED "❌ Algunos nodos no tienen etiquetas GPU"
    SUCCESS=false
fi

# Summary
echo ""
echo -e "${BLUE}=== Resumen de Validación ===${NC}"
if [ "$SUCCESS" = true ]; then
    print_status $GREEN "🎉 ¡Todas las validaciones pasaron!"
    echo ""
    echo "✅ Clúster Kind funcional con:"
    echo "  - 1 control-plane con acceso GPU"
    echo "  - 4 workers con acceso GPU"
    echo "  - NVIDIA device plugin corriendo"
    echo "  - Etiquetas GPU configuradas"
    echo ""
    echo "El clúster está listo para workloads con GPU!"
else
    print_status $RED "❌ Algunas validaciones fallaron"
    echo ""
    echo "Revisa los errores arriba para solucionar los problemas."
fi

exit $([ "$SUCCESS" = true ] && echo 0 || echo 1)