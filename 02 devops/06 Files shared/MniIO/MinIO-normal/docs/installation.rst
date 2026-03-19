Installation
============

This section guides you through the installation and initial setup of the MinIO Docker Compose environment.

System Requirements
-------------------

Before proceeding, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows (with WSL2 for Windows)
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **Python**: Version 3.8 or later (for DVC and optional scripts)
- **Disk Space**: At least 10GB free space for MinIO data and Docker images

.. note::
   For Windows users, Docker Desktop with WSL2 backend is recommended for optimal performance.

Downloading the Project
-----------------------

Clone or download the project repository containing the ``docker-compose.yaml`` file:

.. code-block:: bash

   git clone <repository-url>
   cd minio-docker-setup

Alternatively, download the ZIP file and extract it to your desired directory.

Installing Dependencies
-----------------------

1. **Install Docker**:

   Follow the official Docker installation guide for your platform:

   - `Docker for Linux <https://docs.docker.com/engine/install/linux-postinstall/>`_
   - `Docker for macOS <https://docs.docker.com/desktop/install/mac/>`_
   - `Docker for Windows <https://docs.docker.com/desktop/install/windows-install/>`_

2. **Install Docker Compose**:

   Docker Compose is typically included with Docker Desktop. For Linux systems, install it separately:

   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install docker-compose-plugin

3. **Install DVC (Optional)**:

   If you plan to use this setup with Data Version Control:

   .. code-block:: bash

      pip install dvc

4. **Install Python Dependencies for Examples**:

   For running the Python examples in this documentation:

   .. code-block:: bash

      pip install boto3 pandas scikit-learn

Initial Configuration
---------------------

1. **Volume Setup**:

   Create the data directory for MinIO persistence:

   .. code-block:: bash

      sudo mkdir -p /mnt/DVC_tmp/DVC_data
      sudo chown -R $USER:$USER /mnt/DVC_tmp/DVC_data

   .. warning::
      Adjust the path according to your system's mount points. For local development, consider using ``./DVC_data`` instead.

2. **Environment Variables**:

   Review and modify the environment variables in ``docker-compose.yaml``:

   - ``MINIO_ROOT_USER``: Default admin username
   - ``MINIO_ROOT_PASSWORD``: Change this to a secure password in production

3. **Port Configuration**:

   Ensure ports 30706 and 30707 are available on your system. Modify them in ``docker-compose.yaml`` if conflicts exist.

Verification
------------

After installation, verify your setup:

.. code-block:: bash

   docker --version
   docker-compose --version
   python --version

All commands should return version information without errors.