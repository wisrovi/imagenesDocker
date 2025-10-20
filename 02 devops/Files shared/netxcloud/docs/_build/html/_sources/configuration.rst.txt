Configuration
=============

This section covers configuration options for customizing the Nextcloud Docker deployment.

Environment Variables
---------------------

Customize the deployment using the `.env` file. Below is a detailed list of variables:

.. list-table:: Environment Variables
   :header-rows: 1
   :widths: 20 30 50

   * - Variable
     - Default
     - Description
   * - DB_PASSWORD
     - (required)
     - Password for PostgreSQL database. Must be strong.
   * - NEXTCLOUD_DOMAIN
     - localhost
     - Domain for Nextcloud. Used for trusted domains.
   * - ADMIN_USER
     - admin
     - Initial admin username.
   * - ADMIN_PASSWORD
     - (required)
     - Initial admin password.
   * - ONLYOFFICE_JWT_SECRET
     - (required)
     - Secret for OnlyOffice JWT. Generate a random string.
   * - ONLYOFFICE_DOMAIN
     - onlyoffice.localhost
     - Domain for OnlyOffice service.

.. note::
   All variables are mandatory except where noted. Use `openssl rand -hex 32` to generate secrets.

Nginx Configuration
-------------------

The `nginx/conf.d/default.conf` file configures the reverse proxy:

.. code-block:: nginx

   upstream nextcloud-app {
       server app:80;
   }

   server {
       listen 80;
       server_name ${NEXTCLOUD_DOMAIN};
       return 301 https://$host$request_uri;
   }

   server {
       listen 443 ssl;
       server_name ${NEXTCLOUD_DOMAIN};

       ssl_certificate /etc/nginx/certs/fullchain.pem;
       ssl_certificate_key /etc/nginx/certs/privkey.pem;

       client_max_body_size 10G;

       location / {
           proxy_pass http://nextcloud-app;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto https;
           proxy_max_temp_file_size 0;
           proxy_buffering off;
           proxy_request_buffering off;
       }
   }

.. tip::
   Adjust `client_max_body_size` for larger file uploads if needed.

SSL Certificates
----------------

Certificates are generated using OpenSSL in the `openssl/` directory.

Configuration file `openssl/nginx/conf/openssl_wisrovi.cnf`:

.. code-block:: ini

   [ req ]
   default_bits       = 4096
   prompt             = no
   default_md         = sha256
   distinguished_name = dn
   x509_extensions    = v3_req

   [ dn ]
   C  = ES
   ST = Badajoz
   L  = Badajoz
   O  = Wisrovi Rodriguez
   OU = Autor Independiente
   CN = www.sslcert.wisrovi.com
   emailAddress = wisrovi.rodriguez@gmail.com

   [ v3_req ]
   subjectAltName = @alt_names

   [ alt_names ]
   URI.1 = https://es.linkedin.com/in/wisrovi-rodriguez

For production, use `certbot` for Let's Encrypt certificates.

Volumes
-------

Persistent data is stored in Docker volumes:

- `db_data`: PostgreSQL data.
- `nextcloud_data`: Nextcloud files (commented out by default).
- `onlyoffice_data`: OnlyOffice documents.

To inspect volumes:

.. code-block:: bash

   docker volume ls
   docker volume inspect netxcloud_db_data

Advanced Configuration
----------------------

**Nextcloud Settings**

Edit `docker-compose.yaml` to add Nextcloud environment variables:

.. code-block:: yaml

   app:
     environment:
       - NEXTCLOUD_TRUSTED_DOMAINS=example.com,www.example.com

**Database Tuning**

For PostgreSQL tuning, add to `db` service:

.. code-block:: yaml

   db:
     environment:
       - POSTGRES_SHARED_BUFFERS=256MB

**Redis Configuration**

Customize Redis with a config file:

.. code-block:: yaml

   redis:
     volumes:
       - ./redis.conf:/etc/redis/redis.conf

Security Considerations
-----------------------

- Use HTTPS only.
- Regularly update Docker images.
- Implement firewall rules.
- Monitor logs for suspicious activity.

For more security tips, see the `Nextcloud Security Documentation <https://docs.nextcloud.com/server/latest/admin_manual/configuration_server/security_setup_warnings.html>`_.