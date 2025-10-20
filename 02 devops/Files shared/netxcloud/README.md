# Nextcloud Docker Deployment

This project provides a complete Docker-based deployment setup for Nextcloud, a self-hosted file sharing and collaboration platform. It includes all necessary components for a production-ready environment, such as a PostgreSQL database, Redis caching, OnlyOffice document server for online editing, and an Nginx reverse proxy with SSL termination.

## Overview

Nextcloud is an open-source platform that allows users to store, share, and collaborate on files securely. This Docker stack simplifies the deployment process by containerizing all services, ensuring scalability, security, and ease of management.

The setup includes:
- **Nextcloud Application**: Core file sharing and collaboration features.
- **PostgreSQL Database**: Persistent data storage for Nextcloud.
- **Redis**: High-performance caching to improve response times.
- **OnlyOffice Document Server**: Enables online document editing within Nextcloud.
- **Nginx Reverse Proxy**: Handles HTTPS traffic, SSL certificates, and load balancing.
- **Documentation Server**: Sphinx-generated project documentation served via Nginx.
- **SSL Certificate Generation**: Automated creation of self-signed SSL certificates using OpenSSL.

## Features

- **Secure by Default**: HTTPS enforced with automatic SSL certificate generation.
- **Scalable Architecture**: Modular services that can be scaled independently.
- **Data Persistence**: Volumes for database, Nextcloud data, and OnlyOffice data.
- **High Availability**: Redis caching reduces load on the database.
- **Document Collaboration**: Integrated OnlyOffice for real-time document editing.
- **Easy Configuration**: Environment variables for customization.
- **Production Ready**: Includes best practices for security and performance.

## Prerequisites

Before deploying, ensure you have the following installed on your system:

- **Docker**: Version 20.10 or later.
- **Docker Compose**: Version 2.0 or later.
- **Git**: For cloning the repository (optional).
- **Domain Name**: A domain pointing to your server's IP (recommended for SSL).

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd nextcloud-docker-deployment
   ```

2. **Create Environment File**:
   Create a `.env` file in the root directory with the following variables:
   ```env
   DB_PASSWORD=your_secure_db_password
   NEXTCLOUD_DOMAIN=your-domain.com
   ADMIN_USER=admin
   ADMIN_PASSWORD=your_admin_password
   ONLYOFFICE_JWT_SECRET=your_jwt_secret
   ONLYOFFICE_DOMAIN=onlyoffice.your-domain.com
   ```

   - `DB_PASSWORD`: Password for the PostgreSQL database.
   - `NEXTCLOUD_DOMAIN`: Domain for Nextcloud (e.g., `nextcloud.example.com`).
   - `ADMIN_USER` and `ADMIN_PASSWORD`: Initial admin credentials for Nextcloud.
   - `ONLYOFFICE_JWT_SECRET`: Secret key for secure communication between Nextcloud and OnlyOffice.
   - `ONLYOFFICE_DOMAIN`: Domain for OnlyOffice (if using a separate domain).

3. **Generate SSL Certificates**:
   Navigate to the `openssl/` directory and run:
   ```bash
   docker-compose up
   ```
   This will generate `fullchain.pem` and `privkey.pem` in the `certs/` directory.

4. **Start the Stack**:
   From the root directory, run:
   ```bash
   docker-compose up -d
   ```
   This will pull images, build containers, and start all services in detached mode.

5. **Access Nextcloud**:
   Open your browser and navigate to `https://your-domain.com`. Log in with the admin credentials specified in the `.env` file.

## Configuration

### Environment Variables

Customize the deployment using the `.env` file. Key variables include:

- Database settings: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.
- Nextcloud settings: `NEXTCLOUD_TRUSTED_DOMAINS`, `NEXTCLOUD_ADMIN_USER`, `NEXTCLOUD_ADMIN_PASSWORD`.
- OnlyOffice settings: `JWT_SECRET`, `VIRTUAL_HOST`.
- Nginx settings: Handled via `nginx/conf.d/default.conf`.

### Nginx Configuration

The `nginx/conf.d/default.conf` file configures the reverse proxy:
- Redirects HTTP to HTTPS.
- Serves SSL certificates from `/etc/nginx/certs`.
- Allows large file uploads (up to 10GB).
- Proxies requests to the Nextcloud app container.

### SSL Certificates

SSL certificates are generated using a separate Docker container in the `openssl/` directory:
- Uses OpenSSL to create a self-signed certificate valid for 825 days.
- Configuration file: `openssl/nginx/conf/openssl_wisrovi.cnf`.
- Certificates are stored in `certs/` and mounted into the Nginx container.

For production, consider using Let's Encrypt or a trusted CA instead of self-signed certificates.

### Volumes

Persistent data is stored in Docker volumes:
- `db_data`: PostgreSQL database files.
- `nextcloud_data`: Nextcloud application data (currently commented out in docker-compose.yaml).
- `onlyoffice_data`: OnlyOffice document data.

## Services

### Database (PostgreSQL)
- Image: `postgres:15-alpine`
- Stores user data, files metadata, and configurations.

### Cache (Redis)
- Image: `redis:alpine`
- Improves performance by caching sessions and data.

### Nextcloud App
- Image: `nextcloud:latest`
- Core application handling file operations, user management, and integrations.

### OnlyOffice
- Image: `onlyoffice/documentserver:latest`
- Provides online document editing capabilities.

### Nginx Proxy
- Image: `nginx:alpine`
- Handles incoming traffic, SSL, and routing to services.

### Documentation Server
- Built from `./docs/Dockerfile`
- Serves Sphinx-generated HTML documentation on port 8080.

## Usage

- **File Management**: Upload, share, and organize files through the Nextcloud web interface.
- **Collaboration**: Use OnlyOffice to edit documents collaboratively.
- **Admin Panel**: Access settings via the admin user to configure apps, users, and security.
- **Documentation**: Access the project documentation at `http://localhost:8080`.
- **Backup**: Regularly back up volumes (`db_data`, `nextcloud_data`, `onlyoffice_data`) for data safety.

## Troubleshooting

- **Port Conflicts**: Ensure ports 80 and 443 are available.
- **SSL Issues**: Verify certificate paths in `nginx/conf.d/default.conf`.
- **Database Connection**: Check environment variables for correct DB credentials.
- **Logs**: Use `docker-compose logs <service>` to view service logs.

## Security Considerations

- Change default passwords in production.
- Use strong, unique secrets for JWT and database.
- Regularly update Docker images for security patches.
- Consider firewall rules to restrict access.
- For public deployments, use valid SSL certificates from a CA.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure all changes are tested and documented.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues or questions, please open an issue on the GitHub repository or refer to the official Nextcloud documentation at [nextcloud.com](https://nextcloud.com).