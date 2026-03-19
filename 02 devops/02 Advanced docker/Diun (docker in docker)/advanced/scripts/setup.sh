#!/bin/bash

# Setup script for Docker-in-Docker project
# This script initializes the project environment

set -e

echo "🚀 Setting up Docker-in-Docker project..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting."; exit 1; }

echo "✅ Prerequisites check passed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p volumes/files volumes/dind-data

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh

# Build documentation
echo "📚 Building documentation..."
cd docs
make html
cd ..

# Generate .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cat > .env << EOF
# Environment variables for Docker-in-Docker project
SSH_PASSWORD=changeme123
POSTGRES_PASSWORD=changeme456
DOCKER_TLS_CERTDIR=
EOF
    echo "⚠️  Please edit .env file and change default passwords!"
fi

# Build and start containers
echo "🐳 Building and starting containers..."
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🏥 Running health checks..."

# Check DinD
if docker-compose exec -T dind docker ps >/dev/null 2>&1; then
    echo "✅ DinD is healthy"
else
    echo "❌ DinD health check failed"
    exit 1
fi

# Check Portainer
if curl -s -f http://localhost:9003 >/dev/null; then
    echo "✅ Portainer is accessible"
else
    echo "❌ Portainer is not accessible"
fi

# Check Docs
if curl -s -f http://localhost:8082 >/dev/null; then
    echo "✅ Documentation is accessible"
else
    echo "❌ Documentation is not accessible"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Access points:"
echo "   - Portainer: http://localhost:9003"
echo "   - Documentation: http://localhost:8082"
echo "   - SSH: ssh root@localhost -p 50422"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
echo ""
echo "⚠️  Remember to change default passwords in .env file!"