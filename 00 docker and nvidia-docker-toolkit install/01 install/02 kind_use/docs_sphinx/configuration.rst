Configuration
=============

Kind Configuration File
-----------------------

The ``kind-config.yaml`` file defines the cluster configuration:

.. literalinclude:: ../config/kind-config.yaml
   :language: yaml

This configuration includes:

- A control-plane node with port mappings to expose services
- Extra mounts for NVIDIA GPU support
- Four worker nodes with GPU support

.. tabs::

   .. tab:: Basic Configuration

      For a simple single-node cluster:

      .. code-block:: yaml

         kind: Cluster
         apiVersion: kind.x-k8s.io/v1alpha4
         nodes:
         - role: control-plane
           kubeadmConfigPatches:
           - |
             kind: InitConfiguration
             nodeRegistration:
               kubeletExtraArgs:
                 node-labels: "ingress-ready=true"
           extraPortMappings:
           - containerPort: 80
             hostPort: 80
             protocol: TCP
           - containerPort: 443
             hostPort: 443
             protocol: TCP

   .. tab:: GPU Configuration

      For GPU-enabled clusters:

      .. code-block:: yaml

         kind: Cluster
         apiVersion: kind.x-k8s.io/v1alpha4
         nodes:
         - role: control-plane
           extraMounts:
           - hostPath: /dev/nvidia0
             containerPath: /dev/nvidia0
           - hostPath: /dev/nvidiactl
             containerPath: /dev/nvidiactl
           - hostPath: /dev/nvidia-uvm
             containerPath: /dev/nvidia-uvm
           - hostPath: /dev/nvidia-modeset
             containerPath: /dev/nvidia-modeset
         - role: worker
           gpu: true
           extraMounts:
           - hostPath: /dev/nvidia0
             containerPath: /dev/nvidia0
           - hostPath: /dev/nvidiactl
             containerPath: /dev/nvidiactl
           - hostPath: /dev/nvidia-uvm
             containerPath: /dev/nvidia-uvm
           - hostPath: /dev/nvidia-modeset
             containerPath: /dev/nvidia-modeset

   .. tab:: Multi-Node Configuration

      For production-like multi-node clusters:

      .. code-block:: yaml

         kind: Cluster
         apiVersion: kind.x-k8s.io/v1alpha4
         nodes:
         - role: control-plane
         - role: worker
         - role: worker
         - role: worker
         - role: worker

Creating the Cluster
--------------------

Create the cluster using the configuration:

.. code-block:: bash

   kind create cluster --config config/kind-config.yaml

Verify that the cluster is running:

.. code-block:: bash

   kubectl get nodes

GPU Configuration
-----------------

For GPU support, ensure NVIDIA drivers are installed on the host and devices are properly mounted.

Install the GPU plugin for Kubernetes:

.. code-block:: bash

   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

Verify available GPUs:

.. code-block:: bash

   kubectl get nodes -o json | jq '.items[].status.capacity'

ArgoCD Configuration
--------------------

Install ArgoCD in the cluster:

.. code-block:: bash

   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

Access the ArgoCD UI:

.. code-block:: bash

   kubectl port-forward svc/argocd-server -n argocd 8080:443

Advanced Configuration
----------------------

.. tabs::

   .. tab:: Ingress Setup

      Configure ingress controller:

      .. code-block:: bash

         kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

   .. tab:: Persistent Storage

      Add local storage class:

      .. code-block:: bash

         kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml

   .. tab:: Monitoring

      Install Prometheus and Grafana:

      .. code-block:: bash

         kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.66.0/bundle.yaml