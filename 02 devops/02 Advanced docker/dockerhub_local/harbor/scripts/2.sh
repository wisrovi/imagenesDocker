#!/bin/bash
# Script to tag an image for Harbor registry
IMAGE=${1:-nginx}
TAG=${2:-latest}
REGISTRY=localhost:40232/library
docker tag $IMAGE:$TAG $REGISTRY/$IMAGE:$TAG