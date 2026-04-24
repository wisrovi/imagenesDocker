#!/bin/bash

# ==========================================
# DEPLOY SCRIPT - PyPI Traffic Generator
# Copia archivos a todos los PCs de la red
# ==========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/pcs.conf"
FILES=("run.sh" "Makefile" "docker_images.tar")

# Leer PCs desde archivo de configuración
PCS=()
while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    PCS+=("$line")
done < "$CONFIG_FILE"

if [ ${#PCS[@]} -eq 0 ]; then
    echo "ERROR: No hay PCs configurados en $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "   PYPI TRAFFIC GENERATOR - DEPLOY"
echo "   PCs: ${#PCS[@]}"
echo "=========================================="

# Verificar que los archivos existen
echo "Verificando archivos locales..."
for file in "${FILES[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$file" ]; then
        echo "ERROR: Archivo $file no encontrado"
        exit 1
    fi
done

echo "Archivos verificados: ${FILES[*]}"

# Copiar a cada PC
for pc in "${PCS[@]}"; do
    echo ""
    echo ">>> Desplegando a $pc..."
    
    # Crear directorio remoto
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$pc" "mkdir -p ~/pypi-bot" 2>/dev/null || {
        echo "ERROR: No se pudo conectar a $pc"
        continue
    }
    
    # Copiar archivos
    scp -o ConnectTimeout=10 "${FILES[@]}" "$pc:~/pypi-bot/" 2>/dev/null
    
    echo "✓ Archivos copiados a $pc"
done

echo ""
echo "=========================================="
echo "   DEPLOY COMPLETADO"
echo "=========================================="