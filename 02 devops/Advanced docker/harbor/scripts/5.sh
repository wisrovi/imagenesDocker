#!/bin/bash
# Script to pull image from Harbor registry
IMAGE=${1:-nginx}
TAG=${2:-latest}
REGISTRY=localhost:40232/library
docker pull $REGISTRY/$IMAGE:$TAG