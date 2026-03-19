Troubleshooting
===============

Common Issues and Solutions
===========================

Cluster Creation Fails
-----------------------

- Ensure Docker is running: ``sudo systemctl start docker``
- Check Docker info: ``docker info``
- If timeout occurs, simplify the configuration (reduce nodes)
- Check available disk space and memory

Nodes Not Ready
---------------

- Wait 30-60 seconds after creation
- Check CNI: ``kubectl get pods -n kube-system``
- Verify network connectivity: ``kubectl get nodes -o wide``

GPU Not Detected
----------------

- Verify host has NVIDIA drivers: ``nvidia-smi``
- Check mounts in configuration
- Ensure device plugin is running: ``kubectl get pods -n kube-system | grep nvidia``

Port Exposure Issues
--------------------

- Ports 12741-12761 are exposed on control-plane
- Use ``docker ps`` to verify mappings
- Check firewall settings

Device Plugin Errors
--------------------

- "too many open files": Increase ulimits or skip the plugin
- Use direct mounts for GPU access
- Check plugin logs: ``kubectl logs -n kube-system nvidia-device-plugin-daemonset-*``

.. tabs::

   .. tab:: GPU Issues

      **Problem**: GPU not available in pods

      **Solution**:

      .. code-block:: bash

         # Check GPU devices on host
         ls -la /dev/nvidia*

         # Verify mounts in kind config
         cat config/kind-config.yaml

         # Restart device plugin
         kubectl delete pod -n kube-system -l app=nvidia-device-plugin-daemonset
         kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

   .. tab:: Network Issues

      **Problem**: Pods cannot communicate

      **Solution**:

      .. code-block:: bash

         # Check CNI pods
         kubectl get pods -n kube-system

         # Restart CNI
         kubectl delete pod -n kube-system -l k8s-app=kube-proxy
         kubectl delete pod -n kube-system -l k8s-app=kube-dns

   .. tab:: Resource Issues

      **Problem**: Insufficient resources

      **Solution**:

      .. code-block:: bash

         # Check resource usage
         kubectl describe nodes

         # Increase Docker resources
         docker system prune -a

         # Check disk space
         df -h

Recreating Cluster
------------------

.. code-block:: bash

   kind delete cluster
   kind create cluster --config config/kind-config.yaml

Diagnostic Commands
-------------------

Useful commands for debugging:

.. code-block:: bash

   # Cluster status
   kind get clusters
   kubectl cluster-info

   # Node details
   kubectl describe nodes

   # Pod logs
   kubectl logs -n kube-system <pod-name>

   # Docker containers
   docker ps -a

   # System resources
   free -h
   df -h

Getting Help
------------

If you continue to have issues:

1. Check the `Kind documentation <https://kind.sigs.k8s.io/docs/>`_
2. Search `Kubernetes issues <https://github.com/kubernetes/kubernetes/issues>`_
3. Ask on `Stack Overflow <https://stackoverflow.com/questions/tagged/kubernetes>`_
4. Create an issue in the project repository