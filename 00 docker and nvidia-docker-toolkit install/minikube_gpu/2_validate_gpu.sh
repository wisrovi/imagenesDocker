#!/bin/bash
# This script checks for available NVIDIA GPUs on the Kubernetes nodes.
# It prints the node name, whether a GPU is available (true/false), and the number of GPUs.

kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | \
  awk '{ if ($2+0 > 0) print $1, "true", $2+0; else print $1, "false", 0 }'
