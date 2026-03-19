SSL Certificates
=================

This section covers SSL certificate management in the ``SSL_certificates/`` directory.

Let's Encrypt
-------------

Automated SSL certificate generation using Let's Encrypt.

Location: ``SSL_certificates/letsencript/``

Usage
~~~~~

1. Modify ``create_dns_01.sh`` with your domain information
2. Run the script:

   .. code-block:: bash

      sudo sh create_dns_01.sh

3. Add the DNS TXT record as instructed
4. Press Enter to complete certificate generation

Certificates are stored in ``nginx/certs/letsencrypt/``

OpenSSL
-------

Manual SSL certificate generation using OpenSSL.

Location: ``SSL_certificates/openssl/``

Usage
~~~~~

.. code-block:: bash

   cd SSL_certificates/openssl/
   docker-compose up -d

This generates self-signed certificates for testing purposes.

Nginx Configuration
~~~~~~~~~~~~~~~~~~~

Example Nginx SSL configuration:

.. code-block:: nginx

   server {
       listen 443 ssl;
       server_name example.com;

       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;

       # Additional SSL settings...
   }