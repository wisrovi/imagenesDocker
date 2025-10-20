Installation
============

This section guides you through the installation process of the Nextcloud Docker stack.

Prerequisites
-------------

Before proceeding, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows (with WSL2).
- **Docker**: Version 20.10 or later. Download from `Docker Official Site <https://www.docker.com/get-started>`_.
- **Docker Compose**: Version 2.0 or later. Included with Docker Desktop.
- **Git**: For cloning the repository. Download from `Git Official Site <https://git-scm.com>`_.
- **Domain Name**: Recommended for SSL. Point it to your server's IP.

System Requirements
~~~~~~~~~~~~~~~~~~~

- **CPU**: At least 2 cores.
- **RAM**: Minimum 4GB, recommended 8GB+.
- **Storage**: 20GB+ free space for data and containers.

Step-by-Step Installation
-------------------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd nextcloud-docker-deployment

Replace `<repository-url>` with the actual repository URL.

2. Create Environment File
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a `.env` file in the root directory:

.. code-block:: env

   DB_PASSWORD=your_secure_db_password
   NEXTCLOUD_DOMAIN=your-domain.com
   ADMIN_USER=admin
   ADMIN_PASSWORD=your_admin_password
   ONLYOFFICE_JWT_SECRET=your_jwt_secret
   ONLYOFFICE_DOMAIN=onlyoffice.your-domain.com

.. warning::
   Use strong, unique passwords. Never commit the `.env` file to version control.

3. Generate SSL Certificates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to the `openssl/` directory and run:

.. code-block:: bash

   docker-compose up

This generates `fullchain.pem` and `privkey.pem` in the `certs/` directory.

For production, consider using `Let's Encrypt <https://letsencrypt.org>`_ for valid certificates.

4. Start the Stack
~~~~~~~~~~~~~~~~~~

From the root directory:

.. code-block:: bash

   docker-compose up -d

This pulls images, builds containers, and starts all services in detached mode.

5. Access Nextcloud
~~~~~~~~~~~~~~~~~~~

Open your browser and navigate to `https://your-domain.com`. Log in with the admin credentials from the `.env` file.

.. tip::
   The first startup may take a few minutes as images are downloaded.

Using Make Commands
-------------------

Alternatively, use the provided Makefile:

.. code-block:: bash

   make up  # Start the stack
   make down  # Stop the stack
   make logs  # View logs

See :doc:`usage` for more commands.

Post-Installation Steps
-----------------------

- **Configure Apps**: Install and configure Nextcloud apps from the admin panel.
- **Set Up Backups**: Regularly back up volumes as described in :doc:`usage`.
- **Monitor Logs**: Use `docker-compose logs -f` to monitor for issues.

Troubleshooting Installation
----------------------------

- **Port Conflicts**: Ensure ports 80, 443, and 8080 are free.
- **Permission Issues**: Run commands with appropriate permissions (e.g., sudo if needed).
- **SSL Errors**: Verify certificate paths in `nginx/conf.d/default.conf`.

If you encounter issues, refer to the :doc:`troubleshooting` section.