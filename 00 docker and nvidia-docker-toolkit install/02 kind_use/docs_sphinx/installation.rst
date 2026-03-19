Installation
============

Prerequisites
-------------

- Docker installed and running
- NVIDIA drivers (if GPU support is required)
- kubectl installed (optional, but recommended)

Installing Kind
---------------

Download and install Kind:

.. code-block:: bash

   curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
   chmod +x ./kind
   sudo mv ./kind /usr/local/bin/kind

Verify the installation:

.. code-block:: bash

   kind --version

Installing Docker
-----------------

Ensure Docker is installed and running:

.. code-block:: bash

   sudo systemctl start docker
   docker info

Installing kubectl
------------------

Download kubectl:

.. code-block:: bash

   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   chmod +x kubectl
   sudo mv kubectl /usr/local/bin/

Verify:

.. code-block:: bash

   kubectl version --client

Python Dependencies
-------------------

For automation scripts, install Python dependencies:

.. code-block:: bash

   pip install pyyaml kubernetes

Or use the requirements file:

.. code-block:: bash

   pip install -r requirements.txt

Automated Installation
----------------------

Use the provided setup script for automated installation:

.. code-block:: bash

   ./scripts/setup.sh

This script will install all required components automatically.