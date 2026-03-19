#!/bin/bash

# Test script for Docker-in-Docker project
# Runs basic tests to ensure everything is working

set -e

echo "🧪 Running tests for Docker-in-Docker project..."

# Test 1: Check if containers are running
echo "Test 1: Checking container status..."
if [ "$(docker-compose ps | grep -c "Up")" -lt 2 ]; then
    echo "❌ Not all containers are running"
    docker-compose ps
    exit 1
fi
echo "✅ All containers are running"

# Test 2: Check DinD functionality
echo "Test 2: Testing DinD functionality..."
if ! docker-compose exec -T dind docker run --rm hello-world >/dev/null 2>&1; then
    echo "❌ DinD is not working properly"
    exit 1
fi
echo "✅ DinD is working"

# Test 3: Check Portainer accessibility
echo "Test 3: Testing Portainer accessibility..."
if ! curl -s -f http://localhost:9003 >/dev/null; then
    echo "❌ Portainer is not accessible"
    exit 1
fi
echo "✅ Portainer is accessible"

# Test 4: Check documentation accessibility
echo "Test 4: Testing documentation accessibility..."
if ! curl -s -f http://localhost:8082 >/dev/null; then
    echo "❌ Documentation is not accessible"
    exit 1
fi
echo "✅ Documentation is accessible"

# Test 5: Check SSH port is open
echo "Test 5: Testing SSH port..."
if ! nc -z localhost 50422 >/dev/null 2>&1; then
    echo "❌ SSH port is not open"
    exit 1
fi
echo "✅ SSH port is open"

# Test 6: Check volume persistence
echo "Test 6: Testing volume persistence..."
docker-compose exec -T dind touch /app/test_file
if [ ! -f volumes/files/test_file ]; then
    echo "❌ Volume persistence is not working"
    exit 1
fi
docker-compose exec -T dind rm /app/test_file
echo "✅ Volume persistence is working"

echo ""
echo "🎉 All tests passed! The project is working correctly."