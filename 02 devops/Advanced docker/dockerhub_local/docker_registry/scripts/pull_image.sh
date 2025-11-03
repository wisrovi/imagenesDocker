#!/bin/bash

# Script para descargar una imagen del registry local
# Uso: ./pull_image.sh <imagen>

if [ $# -ne 1 ]; then
    echo "Uso: $0 <imagen>"
    echo "Ejemplo: $0 localhost:40231/hello-world"
    exit 1
fi

IMAGE=$1

echo "Descargando imagen $IMAGE..."

if ! docker pull $IMAGE; then
    echo "Error: Falló la descarga de la imagen $IMAGE."
    exit 1
fi

echo "Imagen descargada exitosamente."