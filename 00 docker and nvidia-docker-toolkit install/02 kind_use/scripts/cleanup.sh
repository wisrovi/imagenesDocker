#!/bin/bash

# Cleanup script

echo "Deleting cluster..."
kind delete cluster

echo "Removing Kind binary..."
sudo rm /usr/local/bin/kind

echo "Cleanup complete!"