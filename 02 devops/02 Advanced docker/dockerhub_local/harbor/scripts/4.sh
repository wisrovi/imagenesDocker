#!/bin/bash
# Script to remove the local image after pushing
IMAGE=${1:-nginx}
TAG=${2:-latest}
docker rmi $IMAGE:$TAG