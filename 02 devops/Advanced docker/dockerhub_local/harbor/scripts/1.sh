#!/bin/bash
# Script to pull an image from Docker Hub
IMAGE=${1:-nginx}
TAG=${2:-latest}
docker pull $IMAGE:$TAG