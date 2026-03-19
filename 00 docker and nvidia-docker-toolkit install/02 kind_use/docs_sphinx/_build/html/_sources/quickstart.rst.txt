Quick Start
===========

Get up and running with a Kind cluster with GPU support in minutes.

Prerequisites
-------------

Before starting, ensure you have the following installed:

- Docker (running)
- NVIDIA drivers (if using GPU)
- kubectl (recommended)

One-Command Setup
-----------------

For a quick setup, use the provided automation script:

.. code-block:: bash

   ./scripts/setup.sh

This script will:

1. Install Kind if not present
2. Create the cluster with GPU support
3. Install necessary Kubernetes components
4. Set up ArgoCD for GitOps

Manual Setup
------------

If you prefer manual setup:

1. Install Kind:

   .. code-block:: bash

      curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
      chmod +x ./kind
      sudo mv ./kind /usr/local/bin/kind

2. Create the cluster:

   .. code-block:: bash

      kind create cluster --config config/kind-config.yaml

3. Verify the cluster:

   .. code-block:: bash

      kubectl get nodes

Deploy a GPU Pod
----------------

Test your GPU setup with a simple pod:

.. code-block:: bash

   kubectl apply -f examples/gpu-pod-example.yaml

Check the pod status:

.. code-block:: bash

   kubectl get pods
   kubectl logs gpu-pod-example

Next Steps
----------

- Explore :doc:`configuration` for advanced setup options
- Check out :doc:`examples` for Python automation scripts
- Refer to :doc:`troubleshooting` if you encounter issues