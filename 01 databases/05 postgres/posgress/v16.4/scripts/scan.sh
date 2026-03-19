#!/bin/bash

# Security scan script using Trivy
# Scans the custom PostgreSQL image for vulnerabilities

IMAGE_NAME="wisrovi/postgres:v16.4"

echo "Scanning image $IMAGE_NAME for vulnerabilities..."

docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image \
  --format table \
  --exit-code 0 \
  $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Scan completed successfully."
else
    echo "Scan failed."
    exit 1
fi