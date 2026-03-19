Examples
========

This section provides practical examples and Python scripts for automating Kind cluster management.

Python Automation Scripts
-------------------------

Cluster Management Script
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../scripts/cluster_manager.py
   :language: python
   :lines: 1-50

GPU Resource Monitor
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../scripts/gpu_monitor.py
   :language: python
   :lines: 1-30

Configuration Generator
~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Basic Configuration

      .. code-block:: python

         import yaml

         def generate_basic_config():
             config = {
                 'kind': 'Cluster',
                 'apiVersion': 'kind.x-k8s.io/v1alpha4',
                 'nodes': [
                     {
                         'role': 'control-plane',
                         'kubeadmConfigPatches': [
                             'kind: InitConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: "ingress-ready=true"'
                         ],
                         'extraPortMappings': [
                             {'containerPort': 80, 'hostPort': 80, 'protocol': 'TCP'},
                             {'containerPort': 443, 'hostPort': 443, 'protocol': 'TCP'}
                         ]
                     }
                 ]
             }
             return config

   .. tab:: GPU Configuration

      .. code-block:: python

         import yaml

         def generate_gpu_config():
             config = {
                 'kind': 'Cluster',
                 'apiVersion': 'kind.x-k8s.io/v1alpha4',
                 'nodes': [
                     {
                         'role': 'control-plane',
                         'extraMounts': [
                             {'hostPath': '/dev/nvidia0', 'containerPath': '/dev/nvidia0'},
                             {'hostPath': '/dev/nvidiactl', 'containerPath': '/dev/nvidiactl'},
                             {'hostPath': '/dev/nvidia-uvm', 'containerPath': '/dev/nvidia-uvm'},
                             {'hostPath': '/dev/nvidia-modeset', 'containerPath': '/dev/nvidia-modeset'}
                         ]
                     },
                     {
                         'role': 'worker',
                         'gpu': True,
                         'extraMounts': [
                             {'hostPath': '/dev/nvidia0', 'containerPath': '/dev/nvidia0'},
                             {'hostPath': '/dev/nvidiactl', 'containerPath': '/dev/nvidiactl'},
                             {'hostPath': '/dev/nvidia-uvm', 'containerPath': '/dev/nvidia-uvm'},
                             {'hostPath': '/dev/nvidia-modeset', 'containerPath': '/dev/nvidia-modeset'}
                         ]
                     }
                 ]
             }
             return config

Deployment Examples
-------------------

TensorFlow GPU Pod
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   apiVersion: v1
   kind: Pod
   metadata:
     name: tensorflow-gpu-pod
   spec:
     restartPolicy: OnFailure
     containers:
     - name: tensorflow
       image: tensorflow/tensorflow:latest-gpu
       command: ["python", "-c", "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"]
       resources:
         limits:
           nvidia.com/gpu: 1

PyTorch GPU Pod
~~~~~~~~~~~~~~~

.. code-block:: yaml

   apiVersion: v1
   kind: Pod
   metadata:
     name: pytorch-gpu-pod
   spec:
     restartPolicy: OnFailure
     containers:
     - name: pytorch
       image: pytorch/pytorch:latest
       command: ["python", "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"]
       resources:
         limits:
           nvidia.com/gpu: 1

ArgoCD Application
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   apiVersion: argoproj.io/v1alpha1
   kind: Application
   metadata:
     name: my-app
     namespace: argocd
   spec:
     project: default
     source:
       repoURL: https://github.com/my-org/my-repo
       targetRevision: HEAD
       path: .
     destination:
       server: https://kubernetes.default.svc
       namespace: default
     syncPolicy:
       automated:
         prune: true
         selfHeal: true