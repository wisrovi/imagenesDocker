Installation
============

1. **Clone the Repository** (optional):

   .. code-block:: bash

      git clone <repository-url>
      cd ollama-docker-setup

2. **Ensure Prerequisites**:

   - Install Docker and Docker Compose
   - If using GPU: Install NVIDIA Container Toolkit and verify GPU access

3. **Build Custom Images** (optional):

   .. code-block:: bash

      make build
      # or
      docker-compose build