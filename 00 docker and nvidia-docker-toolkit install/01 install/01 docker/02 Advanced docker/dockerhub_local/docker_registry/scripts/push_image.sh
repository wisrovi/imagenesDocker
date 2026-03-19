#!/bin/bash

# Script para subir una imagen al registry local
# Uso: ./push_image.sh <imagen_origen> <nombre_destino>

if [ $# -ne 2 ]; then
    echo "Uso: $0 <imagen_origen> <nombre_destino>"
    echo "Ejemplo: $0 hello-world localhost:40231/hello-world"
    exit 1
fi

SOURCE_IMAGE=$1
DEST_IMAGE=$2

echo "Subiendo imagen $SOURCE_IMAGE a $DEST_IMAGE..."

# Verificar si la imagen origen existe
if ! docker image inspect $SOURCE_IMAGE > /dev/null 2>&1; then
    echo "Error: La imagen $SOURCE_IMAGE no existe localmente."
    exit 1
fi

# Taggear la imagen
if ! docker tag $SOURCE_IMAGE $DEST_IMAGE; then
    echo "Error: Falló el tagging de la imagen."
    exit 1
fi

# Subir al registry
if ! docker push $DEST_IMAGE; then
    echo "Error: Falló la subida al registry."
    exit 1
fi

echo "Imagen subida exitosamente."