#!/bin/bash

# Script para pruebas de integración: push, pull, list

echo "Iniciando pruebas de integración..."

# 1. Push imagen
echo "1. Subiendo imagen de prueba..."
if ./push_image.sh alpine localhost:40231/alpine:integration-test; then
    echo "   ✓ Push exitoso"
else
    echo "   ✗ Push falló"
    exit 1
fi

# 2. List imágenes
echo "2. Listando imágenes..."
if ./list_images.sh | grep -q "alpine"; then
    echo "   ✓ Alpine en lista"
else
    echo "   ✗ Alpine no en lista"
    exit 1
fi

# 3. Pull imagen
echo "3. Descargando imagen..."
docker rmi localhost:40231/alpine:integration-test > /dev/null 2>&1
if ./pull_image.sh localhost:40231/alpine:integration-test; then
    echo "   ✓ Pull exitoso"
else
    echo "   ✗ Pull falló"
    exit 1
fi

# 4. Limpiar
echo "4. Limpiando imagen de prueba..."
docker rmi localhost:40231/alpine:integration-test > /dev/null 2>&1
echo "   ✓ Limpieza completada"

echo "Pruebas de integración completadas exitosamente."