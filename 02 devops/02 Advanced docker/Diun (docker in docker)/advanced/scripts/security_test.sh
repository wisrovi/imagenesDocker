#!/bin/bash

# Security Tests for Docker-in-Docker project
# Tests security configurations and vulnerabilities

set -e

echo "🔒 Running security tests..."

# Test 1: Check for exposed secrets in environment
echo "Test 1: Checking for exposed secrets..."
if grep -r "password\|secret\|key" .env >/dev/null 2>&1; then
    echo "⚠️  Potential secrets found in .env file"
else
    echo "✅ No obvious secrets in .env"
fi

# Test 2: Check Docker security configurations
echo "Test 2: Checking Docker security..."
if docker-compose exec -T dind docker info | grep -q "Security Options"; then
    echo "✅ Docker security options configured"
else
    echo "⚠️  Docker security options not found"
fi

# Test 3: Check firewall status
echo "Test 3: Checking firewall configuration..."
if docker-compose exec -T dind ufw status | grep -q "Status: active"; then
    echo "✅ Firewall is active"
else
    echo "⚠️  Firewall is not active"
fi

# Test 4: Check SSL/TLS configuration
echo "Test 4: Checking SSL/TLS configuration..."
if curl -k https://localhost:8443 >/dev/null 2>&1; then
    echo "✅ SSL endpoint is accessible"
else
    echo "⚠️  SSL endpoint not accessible"
fi

# Test 5: Check for outdated packages
echo "Test 5: Checking for package updates..."
OUTDATED=$(docker-compose exec -T dind apk version | grep -c "<")
if [ "$OUTDATED" -gt 0 ]; then
    echo "⚠️  $OUTDATED packages are outdated"
else
    echo "✅ All packages are up to date"
fi

# Test 6: Check file permissions
echo "Test 6: Checking file permissions..."
if [ -w /var/log ] && [ -r /etc/ssl ]; then
    echo "✅ File permissions look reasonable"
else
    echo "⚠️  File permissions may need review"
fi

# Test 7: Check for open ports
echo "Test 7: Checking for unexpected open ports..."
OPEN_PORTS=$(docker-compose exec -T dind netstat -tln | grep LISTEN | wc -l)
EXPECTED_PORTS=6  # Adjust based on your configuration
if [ "$OPEN_PORTS" -le "$EXPECTED_PORTS" ]; then
    echo "✅ Number of open ports is within expected range"
else
    echo "⚠️  More open ports than expected: $OPEN_PORTS"
fi

# Test 8: Check SSH security
echo "Test 8: Checking SSH security..."
SSH_CONFIG=$(docker-compose exec -T dind sshd -T 2>/dev/null)
if echo "$SSH_CONFIG" | grep -q "PermitRootLogin no"; then
    echo "⚠️  Root login is disabled (good)"
elif echo "$SSH_CONFIG" | grep -q "PermitRootLogin yes"; then
    echo "⚠️  Root login is enabled (review for production)"
fi

# Test 9: Check for SUID binaries
echo "Test 9: Checking for SUID binaries..."
SUID_COUNT=$(docker-compose exec -T dind find / -perm -4000 2>/dev/null | wc -l)
if [ "$SUID_COUNT" -lt 50 ]; then
    echo "✅ SUID binary count is reasonable"
else
    echo "⚠️  High number of SUID binaries: $SUID_COUNT"
fi

# Test 10: Check container capabilities
echo "Test 10: Checking container capabilities..."
CAPS=$(docker-compose exec -T dind capsh --print | grep -c "cap_")
if [ "$CAPS" -lt 20 ]; then
    echo "✅ Container capabilities are limited"
else
    echo "⚠️  Container has many capabilities: $CAPS"
fi

echo ""
echo "🎯 Security tests completed!"
echo "Review any warnings above and consider implementing additional security measures for production."