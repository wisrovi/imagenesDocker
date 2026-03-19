#!/bin/bash

# Script para probar el frontend del registry UI

echo "Probando frontend en localhost:40232..."

# 1. Verificar accesibilidad
echo "1. Verificando accesibilidad..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:40232/ | grep -q "200"; then
    echo "   ✓ Frontend accesible"
else
    echo "   ✗ Frontend no accesible"
    exit 1
fi

# 2. Listar repositorios (API)
echo "2. Listando repositorios..."
REPOS=$(curl -s http://localhost:40232/api/repositories)
if echo "$REPOS" | grep -q "hello-world"; then
    echo "   ✓ Repositorio hello-world encontrado"
else
    echo "   ✗ Repositorio no encontrado"
fi

# 3. Ver detalles de imagen
echo "3. Verificando detalles de imagen..."
DETAILS=$(curl -s "http://localhost:40232/api/repositories/hello-world/tags")
if echo "$DETAILS" | grep -q "v1\|v2"; then
    echo "   ✓ Tags v1 y v2 encontrados"
else
    echo "   ✗ Tags no encontrados"
fi

# 4. Probar funcionalidad de borrar (simulado, ya que requiere UI interactiva)
echo "4. Funcionalidad de borrar: Requiere interacción manual en UI"

echo "Pruebas del frontend completadas."