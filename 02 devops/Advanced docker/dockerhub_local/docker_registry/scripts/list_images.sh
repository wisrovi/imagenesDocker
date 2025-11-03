#!/bin/bash

# Script para listar imágenes en el registry local

echo "Listando imágenes en el registry..."

if ! curl -s http://localhost:40231/v2/_catalog | jq .; then
    echo "Error: Falló la consulta al registry."
    exit 1
fi

echo "Listado completado."