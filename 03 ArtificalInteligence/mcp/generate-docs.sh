#!/bin/bash

# Script para generar documentación automáticamente

set -e

echo "📚 Generando documentación..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado. Instálalo para generar documentación."
    exit 1
fi

# Instalar dependencias
echo "📦 Instalando dependencias..."
cd docs
pip install -r requirements.txt

# Generar documentación
echo "🔨 Generando documentación HTML..."
make html

echo "✅ Documentación generada en docs/_build/html/"
echo "🌐 Abre docs/_build/html/index.html en tu navegador"

# Opcional: Generar otros formatos
echo "📄 Generando documentación PDF (opcional)..."
make latexpdf 2>/dev/null || echo "⚠️  LaTeX no disponible, saltando PDF"

echo "🎉 ¡Documentación lista!"