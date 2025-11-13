# FAQ

## General

**Q: Why Kind?**
A: Kind provides lightweight Kubernetes clusters for development/testing with GPU support.

**Q: Why not minikube?**
A: Kind supports multi-node clusters better and GPU passthrough.

## Setup

**Q: Cluster creation times out?**
A: Reduce nodes in config or increase timeout. Use 1 control-plane + fewer workers.

**Q: Docker not found?**
A: Ensure Docker daemon is running: `sudo systemctl start docker`

**Q: GPU not detected?**
A: Check host drivers with `nvidia-smi`. Ensure mounts in config match host paths.

## GPU

**Q: Device plugin fails?**
A: Use direct mounts for access. Plugin requires NVIDIA drivers in nodes (not present).

**Q: How to schedule GPU pods?**
A: Use nodeSelector: `nvidia.com/gpu.present: "true"`

**Q: Multiple GPUs?**
A: Config mounts /dev/nvidia0; adjust for more GPUs.

## Usage

**Q: How to access services?**
A: Use exposed ports 12741-12761 on localhost.

**Q: Persistent storage?**
A: Use hostPath volumes or add CSI drivers.

**Q: Add more nodes?**
A: Not supported; recreate cluster with new config.

## Troubleshooting

**Q: Nodes not Ready?**
A: Wait 1-2 minutes. Check CNI pods: `kubectl get pods -n kube-system`

**Q: kubectl connection refused?**
A: Run `kind export kubeconfig` or check cluster status with `kind get clusters`