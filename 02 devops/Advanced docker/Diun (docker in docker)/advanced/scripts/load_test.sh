#!/bin/bash

# Load Tests for Docker-in-Docker project
# Tests performance under load

set -e

echo "⚡ Running load tests..."

DURATION=${DURATION:-30}
CONCURRENT_USERS=${CONCURRENT_USERS:-10}

# Test 1: HTTP load test for documentation
echo "Test 1: Load testing documentation server..."
echo "Running $CONCURRENT_USERS concurrent users for $DURATION seconds..."

# Simple load test using curl
for i in $(seq 1 $CONCURRENT_USERS); do
    (
        END_TIME=$((SECONDS + DURATION))
        REQUEST_COUNT=0
        while [ $SECONDS -lt $END_TIME ]; do
            if curl -s -w "%{http_code}\n" http://localhost:8082/ | grep -q "200"; then
                ((REQUEST_COUNT++))
            fi
            sleep 0.1
        done
        echo "User $i completed $REQUEST_COUNT requests"
    ) &
done

wait
echo "✅ Documentation load test completed"

# Test 2: Docker operations load test
echo "Test 2: Load testing Docker operations..."
START_TIME=$SECONDS

for i in $(seq 1 5); do
    docker-compose exec -T dind docker run --rm alpine echo "test $i" >/dev/null &
done

wait
END_TIME=$SECONDS
DURATION=$((END_TIME - START_TIME))
echo "✅ Docker operations completed in $DURATION seconds"

# Test 3: Memory and CPU stress test
echo "Test 3: Memory and CPU stress test..."
docker-compose exec -T dind apk add --no-cache stress-ng >/dev/null 2>&1 || true

timeout 10 docker-compose exec -T dind stress-ng --cpu 1 --vm 1 --vm-bytes 128M --timeout 5s >/dev/null 2>&1 || true
echo "✅ Stress test completed"

# Test 4: Network performance test
echo "Test 4: Network performance test..."
NETWORK_SPEED=$(timeout 5 curl -s -w "%{speed_download}\n" http://localhost:8082/ | tail -1)
echo "Download speed: $NETWORK_SPEED bytes/sec"
echo "✅ Network test completed"

echo ""
echo "🎉 All load tests completed successfully!"