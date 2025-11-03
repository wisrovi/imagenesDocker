Quick Start
===========

1. **Clone or navigate to the project directory**::

      cd /path/to/docker-in-docker/basic/ubuntu

2. **Build and start the services**::

      docker-compose up -d

3. **Access the services**:

   - **Portainer**: Open http://localhost:50421 in your browser (default credentials: admin/admin)
   - **SSH**: ``ssh root@localhost -p 50422`` (password: password)
   - **Web Terminal**: Open http://localhost:50423 in your browser

4. **Verify the setup**::

      docker-compose logs -f dind-basic