#!/bin/bash

echo "Verificando prerrequisitos para MCP Inspector..."

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Instálalo desde https://docs.docker.com/get-docker/"
    exit 1
else
    echo "✅ Docker está instalado: $(docker --version)"
fi

# Verificar Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Instálalo desde https://docs.docker.com/compose/install/"
    exit 1
else
    echo "✅ Docker Compose está instalado: $(docker-compose --version)"
fi

# Verificar que Docker esté corriendo
if ! docker info &> /dev/null; then
    echo "❌ Docker no está corriendo. Inicia el servicio de Docker."
    exit 1
else
    echo "✅ Docker está corriendo"
fi

echo "🎉 Todos los prerrequisitos están cumplidos. Puedes ejecutar 'make run' para iniciar el inspector."