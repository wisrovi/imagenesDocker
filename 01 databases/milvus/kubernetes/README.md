# Milvus on Kubernetes

This directory contains the Kubernetes manifests to deploy the entire Milvus stack, mirroring the setup from the Docker Compose environment.

## 1. Prerequisites

- A running Kubernetes cluster (e.g., Docker Desktop, Minikube, or any cloud provider).
- `kubectl` command-line tool installed and configured to connect to your cluster.
- `kustomize` installed. It is usually included with `kubectl`.

## 2. Deployment

All resources will be created within the `milvus` namespace.

### Applying the Manifests

To deploy all the components, navigate to this `kubernetes` directory and apply the kustomization file:

```bash
cd kubernetes
kubectl apply -k .
```

This command will create:
- The `milvus` namespace.
- PersistentVolumes and PersistentVolumeClaims for stateful services.
- ConfigMaps for Prometheus and Grafana, including dashboards.
- A Secret for Minio credentials.
- StatefulSets for `etcd`, `minio`, `milvus-standalone`, and `prometheus`.
- Deployments for `grafana` and `attu`.
- Services to expose the applications.

### Verifying the Deployment

Check the status of the pods in the `milvus` namespace:

```bash
kubectl get pods -n milvus -w
```

Wait until all pods are in the `Running` state. This might take a few minutes as the images are pulled and the containers start.

## 3. Accessing Services

Services are exposed via `NodePort`. You can access them on your cluster's node IP. If you are running a local cluster (like Docker Desktop or Minikube), this will typically be `localhost`.

| Service           | URL                               | Credentials                 | Purpose                               |
|-------------------|-----------------------------------|-----------------------------|---------------------------------------|
| **Grafana**       | `http://localhost:30000`          | `admin` / `admin`           | Metrics Visualization & Dashboards    |
| **Milvus SDK**    | `localhost:30001`                 | -                           | For Python clients to connect to Milvus |
| **Minio Console** | `http://localhost:30002`          | `minioadmin` / `minioadmin` | Web UI for the Object Storage         |
| **Prometheus**    | `http://localhost:30003`          | -                           | Metrics Collection                    |
| **Attu (UI)**     | `http://localhost:30004`          | -                           | Web-based Management UI for Milvus    |


## 4. Tearing Down

To delete all the resources created by these manifests, run the following command from this directory:

```bash
kubectl delete -k .
```

This will remove all the deployments, services, and config maps, but it will **not** delete the PersistentVolumeClaims or the data on the `hostPath` volumes. To fully clean up, you must also delete the PVCs and the data on the host machine's path (`/mnt/data/*` by default).

```bash
# Delete the PVCs
kubectl delete pvc -n milvus --all

# Manually delete the data on the node(s)
# sudo rm -rf /mnt/data/etcd /mnt/data/minio /mnt/data/milvus
```
