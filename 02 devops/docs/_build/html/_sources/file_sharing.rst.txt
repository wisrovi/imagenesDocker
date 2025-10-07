File Sharing Services
=====================

This section covers file sharing services in the ``Files shared/`` directory.

FTP Server
----------

A basic FTP server for file transfers.

Location: ``Files shared/ftp/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Files\ shared/ftp/
   docker-compose up -d

Access: ``ftp://localhost:21`` (credentials: user/password)

Nextcloud
---------

Nextcloud is a self-hosted file sharing and collaboration platform.

Location: ``Files shared/netxcloud/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Files\ shared/netxcloud/
   docker-compose up -d

Access: ``http://localhost:8080``

Configuration
~~~~~~~~~~~~~

- Nginx proxy configuration
- Persistent data volumes

Samba
-----

Samba provides SMB/CIFS file sharing.

Location: ``Files shared/samba/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Files\ shared/samba/
   docker-compose up -d

Access: ``\\localhost\shared`` (SMB share)

File Browser
------------

A web-based file browser.

Location: ``Files shared/tcp in browser/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Files\ shared/tcp\ in\ browser/
   docker-compose up -d

Access: ``http://localhost:8080``