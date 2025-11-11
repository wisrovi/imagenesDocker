#!/bin/bash

# Script to prepare images for shared loading in workers

IMAGES_DIR="./docker/images"

mkdir -p $IMAGES_DIR

echo "Pulling images..."
docker pull portainer/portainer-ce:latest
docker pull hurlenko/filebrowser:latest
docker pull nginx:latest
docker pull nvidia/cuda:12.2.0-base-ubuntu22.04

echo "Saving images..."
docker save portainer/portainer-ce:latest > $IMAGES_DIR/portainer.tar
docker save hurlenko/filebrowser:latest > $IMAGES_DIR/filebrowser.tar
docker save nginx:latest > $IMAGES_DIR/nginx.tar
docker save nvidia/cuda:12.2.0-base-ubuntu22.04 > $IMAGES_DIR/nvidia.tar

echo "Images prepared in $IMAGES_DIR"