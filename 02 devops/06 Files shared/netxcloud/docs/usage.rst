Usage
=====

This section explains how to use the deployed Nextcloud instance.

Accessing Nextcloud
-------------------

After installation, access Nextcloud at `https://your-domain.com`.

Log in with the admin credentials from `.env`.

File Management
---------------

- **Upload Files**: Drag and drop or use the upload button.
- **Create Folders**: Right-click > New Folder.
- **Share Files**: Select file > Share > Enter recipient.

Collaboration
-------------

Use OnlyOffice for document editing:

1. Open a document.
2. Click "Edit with OnlyOffice".
3. Collaborate in real-time.

.. tip::
   OnlyOffice supports Word, Excel, and PowerPoint formats.

Admin Panel
-----------

Access settings via the admin user:

- Install apps
- Manage users
- Configure security

Go to Settings > Administration.

Backup and Restore
------------------

Regular backups are crucial:

.. code-block:: bash

   # Stop stack
   docker-compose down

   # Backup volumes
   docker run --rm -v netxcloud_db_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/db_backup.tar.gz -C /data .

   # Start stack
   docker-compose up -d

For restore, reverse the process.

.. warning::
   Test backups regularly.

Makefile Commands
-----------------

Use the Makefile for common tasks:

.. code-block:: bash

   make help      # Show available commands
   make up        # Start stack
   make down      # Stop stack
   make logs      # View logs
   make certs     # Generate certificates
   make clean     # Stop and remove volumes

Updating the Stack
------------------

To update images:

.. code-block:: bash

   docker-compose pull
   docker-compose up -d

Check `docker-compose.yaml` for version pins.

Scaling Services
----------------

Scale services as needed:

.. code-block:: bash

   docker-compose up -d --scale app=2

This creates multiple app instances.

Accessing Documentation
-----------------------

The documentation is available at `http://localhost:8080`.

It includes this guide and API references.

Daily Operations
----------------

- Monitor logs daily.
- Check disk space.
- Update apps regularly.

For more, see `Nextcloud User Manual <https://docs.nextcloud.com/server/latest/user_manual/en>`_.