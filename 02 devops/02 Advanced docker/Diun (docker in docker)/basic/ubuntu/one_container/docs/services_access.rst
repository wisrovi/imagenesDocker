Services and Access
===================

Portainer
---------

- **URL**: http://localhost:50421
- **Purpose**: Web-based Docker management interface
- **Default Credentials**: admin / admin (change on first login)
- **Features**: Container management, image registry, network configuration

SSH
---

- **Command**: ``ssh root@localhost -p 50422``
- **Password**: password (or custom via SSH_PASSWORD env var)
- **Purpose**: Direct shell access to the container

Web Terminal (ttyd)
-------------------

- **URL**: http://localhost:50423
- **Purpose**: Browser-based terminal interface
- **Features**: Full terminal functionality in web browser