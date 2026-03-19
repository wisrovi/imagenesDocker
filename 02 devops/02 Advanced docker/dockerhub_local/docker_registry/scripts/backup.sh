#!/bin/bash

# Script para respaldar el directorio registry-data

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/registry_backup_$TIMESTAMP.tar.gz"

echo "Iniciando backup de registry-data..."

# Crear directorio de backups si no existe
mkdir -p $BACKUP_DIR

# Crear backup
if tar -czf $BACKUP_FILE ./registry-data; then
    echo "Backup creado exitosamente: $BACKUP_FILE"
    echo "Tamaño: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "Error: Falló la creación del backup."
    exit 1
fi

# Limpiar backups antiguos (mantener últimos 5)
cd $BACKUP_DIR
ls -t *.tar.gz | tail -n +6 | xargs -r rm --
echo "Backups antiguos limpiados."

echo "Backup completado."