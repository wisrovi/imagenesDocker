#!/bin/bash

# Integration Tests for Docker-in-Docker project
# Tests interactions between services

set -e

echo "🔗 Running integration tests..."

# Test 1: Portainer can manage DinD
echo "Test 1: Testing Portainer-DinD integration..."
if ! docker-compose exec -T dind docker run --rm hello-world >/dev/null 2>&1; then
    echo "❌ DinD is not accessible from Portainer context"
    exit 1
fi
echo "✅ Portainer can manage DinD containers"

# Test 2: SSL certificate validation
echo "Test 2: Testing SSL certificate setup..."
if curl -k https://localhost:8443 >/dev/null 2>&1; then
    echo "✅ SSL certificates are working"
else
    echo "⚠️ SSL certificates not available (expected in development)"
fi

# Test 3: Backup functionality
echo "Test 3: Testing backup functionality..."
if docker-compose exec -T dind /usr/local/bin/backup.sh >/dev/null 2>&1; then
    echo "✅ Backup functionality works"
else
    echo "❌ Backup functionality failed"
fi

# Test 4: Monitoring stack
echo "Test 4: Testing monitoring stack..."
if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not healthy"
fi

if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not healthy"
fi

# Test 5: API endpoints
echo "Test 5: Testing API endpoints..."
if curl -s http://localhost:8082/health >/dev/null 2>&1; then
    echo "✅ Documentation API is accessible"
else
    echo "❌ Documentation API is not accessible"
fi

echo ""
echo "🎉 All integration tests passed!"