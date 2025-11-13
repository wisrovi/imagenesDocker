#!/bin/bash

# Validation script for prerequisites

echo "Validating prerequisites..."

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker not running or not installed"
    exit 1
fi
echo "✅ Docker OK"

# Check NVIDIA
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA GPU/drivers not available"
    exit 1
fi
echo "✅ NVIDIA GPU OK"

# Check sudo
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  Sudo may require password"
fi

# Check if Kind exists
if command -v kind > /dev/null 2>&1; then
    echo "✅ Kind installed: $(kind --version)"
else
    echo "⚠️  Kind not installed, run install_kind.txt"
fi

# Check kubectl
if command -v kubectl > /dev/null 2>&1; then
    echo "✅ kubectl installed"
else
    echo "⚠️  kubectl not installed"
fi

echo "Validation complete!"