#!/bin/bash

# Script para probar el registry

echo "Probando conectividad al registry..."

# Verificar que el registry responda
if curl -s http://localhost:40231/v2/ > /dev/null; then
    echo "Registry está respondiendo."
else
    echo "Error: Registry no responde."
    exit 1
fi

echo "Prueba completada."