#!/bin/bash

IMAGE_NAME="wisrovi/pr_analizer:latest"

echo "Building Docker image: ${IMAGE_NAME}"

docker build -t ${IMAGE_NAME} .

echo "Build complete."
