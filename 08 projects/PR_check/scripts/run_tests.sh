#!/bin/bash

# This script runs the unit tests and coverage report inside a Docker container.
# It ensures that the tests are run in the same environment as the application.

IMAGE_NAME="wisrovi/pr_analizer:latest"

echo "Running unit tests inside a Docker container..."

# Note: Ensure you have a freshly built image with any code changes.
# You can rebuild the image by running: sh scripts/build_image.sh

docker run --rm \
    -w /usr/src/app \
    ${IMAGE_NAME} \
    pytest --cov=pr_analyzer --cov-report term-missing