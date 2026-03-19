# Troubleshooting

## Cluster Creation Fails

- Ensure Docker is running: `sudo systemctl start docker`
- Check Docker info: `docker info`
- If timeout, simplify config (reduce nodes)

## Nodes Not Ready

- Wait 30-60 seconds after creation
- Check CNI: `kubectl get pods -n kube-system`

## GPU Not Detected

- Verify host has NVIDIA drivers: `nvidia-smi`
- Check mounts in config
- For Kubernetes GPU scheduling, device plugin must run

## Port Exposure Issues

- Ports 12741-12761 are exposed on control-plane
- Use `docker ps` to verify mappings

## Device Plugin Errors

- "too many open files": Increase ulimits or skip plugin
- Use direct mounts for GPU access

## Recreate Cluster

```bash
kind delete cluster
kind create cluster --config kind-config.yaml
```