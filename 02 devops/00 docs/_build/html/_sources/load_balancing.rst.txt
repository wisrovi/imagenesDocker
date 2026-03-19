Load Balancing
==============

This section covers the Nginx-based load balancing configurations provided in the ``balanceo_carga_nginx/`` directory.

Overview
--------

The load balancing setup uses Nginx as a reverse proxy to distribute incoming traffic across multiple backend servers. It includes SSL/TLS termination, basic authentication, and support for multiple upstream servers.

Directory Structure
-------------------

.. code-block:: text

   balanceo_carga_nginx/
   ├── certs/                     # SSL certificates
   │   ├── default.crt
   │   ├── default.key
   │   ├── wisrovi.duckdns.org/
   │   └── dhparam.pem
   ├── config/                    # Nginx configuration files
   │   ├── default.conf
   │   ├── nginx-custom.conf
   │   └── default copy.conf
   ├── html/                      # Sample HTML pages
   │   ├── index.1.html
   │   ├── index.2.html
   │   └── index.3.html
   ├── auth/                      # Authentication files
   │   ├── generar.sh
   │   └── htpasswd
   ├── vhost.d/                   # Virtual host configurations
   ├── demo.yaml
   └── docker-compose.yaml

Configuration Files
-------------------

Nginx Configuration
~~~~~~~~~~~~~~~~~~~

The main Nginx configuration is in ``config/default.conf``. It defines:

- Upstream servers for load balancing
- SSL settings
- Proxy settings
- Authentication

SSL Certificates
~~~~~~~~~~~~~~~~

SSL certificates are stored in the ``certs/`` directory. The setup supports:

- Default self-signed certificates
- Let's Encrypt certificates for ``wisrovi.duckdns.org``
- Diffie-Hellman parameters for enhanced security

Authentication
~~~~~~~~~~~~~~

Basic HTTP authentication is configured using files in the ``auth/`` directory:

- ``htpasswd``: Contains user credentials
- ``generar.sh``: Script to generate password hashes

Deployment
----------

To deploy the load balancer:

1. Navigate to the directory:

   .. code-block:: bash

      cd balanceo_carga_nginx/

2. Start the service:

   .. code-block:: bash

      docker-compose up -d

3. Access the load balancer at ``http://localhost`` or ``https://localhost`` (depending on SSL configuration)

Customization
-------------

Upstream Servers
~~~~~~~~~~~~~~~~

Edit the ``config/default.conf`` file to modify upstream server definitions:

.. code-block:: nginx

   upstream backend {
       server backend1.example.com:80;
       server backend2.example.com:80;
   }

SSL Configuration
~~~~~~~~~~~~~~~~~

To use custom SSL certificates:

1. Place your certificate files in the ``certs/`` directory
2. Update the SSL paths in ``config/default.conf``

Authentication
~~~~~~~~~~~~~~

To add or modify users:

1. Run the ``auth/generar.sh`` script
2. Update the ``htpasswd`` file with new credentials

Troubleshooting
---------------

Common issues and solutions:

- **SSL Errors**: Check certificate validity and paths
- **Authentication Failures**: Verify credentials in ``htpasswd``
- **Backend Connection Issues**: Ensure upstream servers are accessible