#!/bin/bash

# Script de configuración inicial del proyecto MCP Inspector

set -e

echo "🚀 Configuración inicial de MCP Inspector"
echo "========================================"

# Verificar prerrequisitos
echo "📋 Verificando prerrequisitos..."
./check-prerequisites.sh

# Crear archivo .env si no existe
if [ ! -f .env ]; then
    echo "📝 Creando archivo de configuración .env..."
    cp .env.example .env
    echo "✅ Archivo .env creado. Edítalo según tus necesidades."
else
    echo "ℹ️  Archivo .env ya existe."
fi

# Generar documentación (opcional)
read -p "❓ ¿Quieres generar la documentación local? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📚 Generando documentación..."
    ./generate-docs.sh
fi

echo ""
echo "🎉 ¡Configuración completa!"
echo ""
echo "Para iniciar el inspector:"
echo "  make run"
echo ""
echo "Para más comandos:"
echo "  make help"