#!/bin/bash

# Script de monitoreo para MCP Inspector

set -e

echo "📊 Monitoreo de MCP Inspector"
echo "================================"

# Verificar si el contenedor está ejecutándose
if docker-compose ps | grep -q "mcp-inspector"; then
    echo "✅ Contenedor ejecutándose"

    # Mostrar estado del contenedor
    echo ""
    echo "Estado del contenedor:"
    docker-compose ps

    # Mostrar uso de recursos
    echo ""
    echo "Uso de recursos:"
    docker stats --no-stream $(docker-compose ps -q mcp-inspector)

    # Mostrar logs recientes
    echo ""
    echo "Logs recientes:"
    docker-compose logs --tail=10 mcp-inspector

    # Verificar conectividad
    echo ""
    echo "Verificando conectividad..."
    if curl -f -s http://localhost:6274 > /dev/null 2>&1; then
        echo "✅ Interfaz web accesible en http://localhost:6274"
    else
        echo "❌ Interfaz web no accesible"
    fi

    if curl -f -s http://localhost:6277 > /dev/null 2>&1; then
        echo "✅ API del servidor accesible en http://localhost:6277"
    else
        echo "⚠️  API del servidor requiere autenticación o no está disponible"
    fi

else
    echo "❌ Contenedor no ejecutándose"
    echo ""
    echo "Para iniciar: make run"
    exit 1
fi

echo ""
echo "================================"