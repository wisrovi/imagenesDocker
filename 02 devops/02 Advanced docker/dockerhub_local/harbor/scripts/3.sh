#!/bin/bash
# Script to push image to Harbor registry
IMAGE=${1:-nginx}
TAG=${2:-latest}
REGISTRY=localhost:40232/library
docker push $REGISTRY/$IMAGE:$TAG