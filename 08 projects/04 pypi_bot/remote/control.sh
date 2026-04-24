#!/bin/bash

# ==========================================
# CONTROL SCRIPT - PyPI Traffic Generator
# Control centralizado de todos los PCs
# ==========================================

# Cargar IPs desde archivo de configuración
CONFIG_FILE="$(dirname "$0")/pcs.conf"
REMOTE_DIR="~/pypi-bot"

# Leer PCs del archivo de configuración (ignora líneas que empiezan con #)
PCS=()
while IFS= read -r line || [ -n "$line" ]; do
    # Ignorar comentarios y líneas vacías
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    PCS+=("$line")
done < "$CONFIG_FILE"

if [ ${#PCS[@]} -eq 0 ]; then
    echo "ERROR: No hay PCs configurados en $CONFIG_FILE"
    exit 1
fi

echo "Cargados ${#PCS[@]} PCs desde $CONFIG_FILE"

usage() {
    echo "Uso: $0 <comando> [número]"
    echo ""
    echo "Comandos:"
    echo "  start <n>    - Iniciar N réplicas en cada PC"
    echo "  stop         - Detener todas las réplicas"
    echo "  status       - Ver estado de todas las réplicas"
    echo "  logs         - Ver logs de todos los PCs"
    echo "  deploy       - Desplegar archivos a todos los PCs"
    echo "  clean        - Borrar todo rastro de los PCs"
    echo ""
    echo "Ejemplos:"
    echo "  $0 start 10    - Iniciar 10 réplicas por PC"
    echo "  $0 stop        - Detener todo"
    echo "  $0 status      - Ver estado"
    echo "  $0 clean       - Borrar todo rastro"
    exit 1
}

deploy_files() {
    echo "=========================================="
    echo "   DESPLIEGUE DE ARCHIVOS"
    echo "=========================================="
    
    for pc in "${PCS[@]}"; do
        echo -n "Desplegando a $pc... "
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "mkdir -p ~/pypi-bot" 2>/dev/null
        scp -o ConnectTimeout=10 run.sh Makefile docker_images.tar "$pc:~/pypi-bot/" 2>/dev/null
        echo "OK"
    done
    echo "=========================================="
}

do_start() {
    local n=${1:-5}
    echo "=========================================="
    echo "   INICIANDO $n REPLICAS POR PC"
    echo "=========================================="
    
    for pc in "${PCS[@]}"; do
        echo ">>> $pc"
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "cd $REMOTE_DIR && make start-n N=$n" 2>/dev/null &
    done
    
    sleep 2
    echo "=========================================="
    echo "Iniciadas $n réplicas en ${#PCS[@]} PCs"
    echo "Total estimado: $((n * ${#PCS[@]})) réplicas"
    echo "=========================================="
}

do_stop() {
    echo "=========================================="
    echo "   DETENIENDO TODAS LAS REPLICAS"
    echo "=========================================="
    
    for pc in "${PCS[@]}"; do
        echo -n "Deteniendo $pc... "
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "pkill -f 'run.sh'" 2>/dev/null
        echo "OK"
    done
    echo "=========================================="
}

do_status() {
    echo "=========================================="
    echo "   ESTADO DE TODOS LOS PCs"
    echo "=========================================="
    
    for pc in "${PCS[@]}"; do
        echo ""
        echo "--- $pc ---"
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "ps aux | grep -v grep | grep run.sh | wc -l" 2>/dev/null || echo "0"
    done
    echo "=========================================="
}

do_logs() {
    echo "=========================================="
    echo "   LOGS EN TIEMPO REAL (Ctrl+C para salir)"
    echo "=========================================="
    
    for pc in "${PCS[@]}"; do
        echo "=== LOGS $pc ===" &
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "tail -f $REMOTE_DIR/monitor_pypi.log" 2>/dev/null &
    done
    wait
}

do_clean() {
    echo "=========================================="
    echo "   LIMPIEZA TOTAL - BORRANDO RASTRO"
    echo "=========================================="
    echo "Este comando:"
    echo "  1. Detiene todas las réplicas"
    echo "  2. Borra archivos del directorio pypi-bot"
    echo "  3. Limpia logs de Docker"
    echo "  4. Elimina imágenes cacheadas"
    echo ""
    read -p "Continuar? (s/n): " confirm
    if [ "$confirm" != "s" ] && [ "$confirm" != "S" ]; then
        echo "Cancelado"
        exit 0
    fi
    
    for pc in "${PCS[@]}"; do
        echo ">>> Limpiando $pc..."
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" '
            pkill -f "run.sh" 2>/dev/null || true
            sleep 1
            rm -rf ~/pypi-bot
            docker system prune -af --volumes 2>/dev/null || true
            docker rmi $(docker images -q) 2>/dev/null || true
        ' 2>/dev/null
        echo "  ✓ $pc limpiado"
    done
    
    echo ""
    echo "=========================================="
    echo "   LIMPIEZA COMPLETADA"
    echo "=========================================="
}

# Main
case "$1" in
    deploy)
        deploy_files
        ;;
    start)
        [ -z "$2" ] && usage
        do_start "$2"
        ;;
    stop)
        do_stop
        ;;
    status)
        do_status
        ;;
    logs)
        do_logs
        ;;
    clean)
        do_clean
        ;;
    *)
        usage
        ;;
esac