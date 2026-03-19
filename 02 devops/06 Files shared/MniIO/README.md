# MinIO Docker Setups

## Overview

This repository provides comprehensive Docker Compose configurations for deploying MinIO, a high-performance, Kubernetes-native object storage server fully compatible with Amazon S3 APIs. The repository includes two distinct setups to cater to different deployment needs:

- **MinIO-Normal**: A basic, non-encrypted setup optimized for development, testing, and local workflows, particularly integrated with Data Version Control (DVC) for machine learning projects.
- **MinIO-SSL**: A secure setup with SSL/TLS encryption enabled, including automated certificate generation, suitable for production environments or secure development scenarios.

MinIO is designed to provide scalable, distributed object storage with features like erasure coding, bitrot protection, and multi-cloud support. These configurations leverage Docker for easy deployment and management.

## Project Structure

```
MniIO/
├── MinIO-normal/
│   ├── README.md                 # Detailed documentation for basic setup
│   └── docker-compose.yaml       # Docker Compose configuration for MinIO server
├── Minio-ssl/
│   ├── README.md                 # Detailed documentation for SSL setup
│   ├── docker-compose.yaml       # Docker Compose configuration for SSL-enabled MinIO
│   ├── openssl/
│   │   ├── docker-compose.yaml   # Certificate generation service
│   │   ├── nginx/
│   │   │   ├── Dockerfile        # OpenSSL container build configuration
│   │   │   └── conf/
│   │   │       └── openssl_wisrovi.cnf  # OpenSSL certificate configuration
│   │   └── Readme.md             # Legacy documentation
│   └── docs/                     # Sphinx-generated documentation
│       ├── _build/               # Built HTML documentation
│       ├── _static/              # Static assets for documentation
│       └── *.rst                 # ReStructuredText source files
└── README.md                     # This file
```

## Prerequisites

Before deploying either setup, ensure your system meets the following requirements:

- **Docker**: Version 20.10 or later (recommended for optimal compatibility)
- **Docker Compose**: Version 2.0 or later
- **Operating System**: Linux, macOS, or Windows with Docker Desktop
- **Disk Space**: Sufficient storage for your data needs (the default configurations mount host directories for persistence)
- **Network**: Access to ports 30706 (MinIO S3 API) and 30707 (MinIO Console) on your host machine

## MinIO-Normal Setup

### Description

The MinIO-Normal setup provides a straightforward, non-encrypted deployment of MinIO suitable for local development, testing, and integration with tools like DVC. This configuration is ideal for machine learning workflows where data versioning and storage are managed through DVC's remote storage capabilities.

### Key Features

- Simple Docker Compose deployment
- Persistent data storage via host volume mounting
- Pre-configured credentials for immediate use
- Integration-ready for DVC and other S3-compatible tools
- Optional default bucket creation

### Installation and Setup

1. **Navigate to the Setup Directory**:
   ```bash
   cd MinIO-normal
   ```

2. **Configure Data Storage**:
   - The default configuration mounts `/mnt/DVC_tmp/DVC_data` on the host to `/data` in the container.
   - If this path doesn't exist or you prefer a different location, modify the `volumes` section in `docker-compose.yaml`:
     ```yaml
     volumes:
       - /path/to/your/data:/data
     ```
   - Alternatively, use a relative path like `./DVC_data:/data` for local storage.

3. **Environment Configuration**:
   - Default credentials are set for convenience:
     - Username: `DVC`
     - Password: `uTAntEMTuVpcJucNjOJm`
   - For production use, modify these in the `environment` section of `docker-compose.yaml`.
   - Optionally, uncomment `MINIO_DEFAULT_BUCKETS=datasets` to create a default bucket on startup.

4. **Start the Service**:
   ```bash
   docker-compose up -d
   ```

   This command will:
   - Pull the MinIO image (if not cached)
   - Start the MinIO server in detached mode
   - Map the required ports
   - Mount the data volume for persistence

### Configuration Details

- **Image**: `minio/minio:RELEASE.2025-02-28T09-55-16Z` (latest stable release at time of configuration)
- **Ports**:
  - `30706:9000` - S3 API endpoint (external port 30706, internal 9000)
  - `30707:9001` - MinIO Console (external port 30707, internal 9001)
- **Environment Variables**:
  - `MINIO_ROOT_USER`: Admin username
  - `MINIO_ROOT_PASSWORD`: Admin password
  - `MINIO_DEFAULT_BUCKETS`: Optional default buckets to create
- **Volumes**:
  - Host data directory mounted to `/data` for persistent storage

### Usage

#### Accessing MinIO

- **S3 API Endpoint**: `http://localhost:30706`
- **Web Console**: `http://localhost:30707`

Log in using the configured credentials (default: DVC / uTAntEMTuVpcJucNjOJm).

#### Integration with DVC

This setup is specifically designed for DVC integration:

1. **Install DVC** (if not already installed):
   ```bash
   pip install dvc
   ```

2. **Configure DVC Remote**:
   ```bash
   dvc remote add -d myremote s3://bucket-name
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm
   ```

3. **Create a Bucket**:
   - Use the MinIO Console to create a bucket, or via DVC commands.

4. **Use in Workflows**:
   ```bash
   dvc add data/file.csv
   dvc push
   ```

#### Basic Operations

- **Stop the Server**:
  ```bash
  docker-compose down
  ```

- **View Logs**:
  ```bash
  docker-compose logs -f dvc-minio
  ```

## MinIO-SSL Setup

### Description

The MinIO-SSL setup provides a secure deployment of MinIO with SSL/TLS encryption enabled. This configuration includes automated generation of SSL certificates using OpenSSL, making it suitable for production environments or any scenario requiring encrypted communication.

### Key Features

- SSL/TLS encryption for all communications
- Automated certificate generation with OpenSSL
- Self-signed certificates (replace with CA-signed for production)
- Secure access to both S3 API and Console
- Persistent encrypted storage

### Installation and Setup

1. **Navigate to the Setup Directory**:
   ```bash
   cd Minio-ssl
   ```

2. **Generate SSL Certificates**:
   ```bash
   cd openssl
   docker-compose up
   ```

   This process will:
   - Build a Docker image with OpenSSL
   - Generate a self-signed SSL certificate valid for 825 days
   - Create the following files in the `certs` directory at the project root:
     - `fullchain.pem`: Public certificate chain
     - `privkey.pem`: Private key

   The certificate is configured for domain `www.dvc.ecapturedtech.com` with subject alternative name `https://ecapturedtech.com/`. Modify `openssl/nginx/conf/openssl_wisrovi.cnf` if you need different domain details.

3. **Start the MinIO Server**:
   ```bash
   cd ..  # Return to Minio-ssl directory
   docker-compose up -d
   ```

   This will start MinIO with SSL enabled, mounting the generated certificates.

### Configuration Details

- **Image**: `minio/minio:RELEASE.2025-02-28T09-55-16Z`
- **Ports**:
  - `30706:9000` - S3 API (HTTPS)
  - `30707:9001` - MinIO Console (HTTPS)
- **Environment Variables**:
  - `MINIO_ROOT_USER`: Admin username (default: DVC)
  - `MINIO_ROOT_PASSWORD`: Admin password (default: uTAntEMTuVpcJucNjOJm)
  - `MINIO_CERT_PUBLIC_KEY`: Path to public certificate
  - `MINIO_CERT_PRIVATE_KEY`: Path to private key
- **Volumes**:
  - `./DVC_data:/data`: Persistent data storage
  - `./certs:/root/.minio/certs`: SSL certificates

### Usage

#### Accessing MinIO

- **S3 API Endpoint**: `https://localhost:30706`
- **Web Console**: `https://localhost:30707`

**Note**: Since certificates are self-signed, browsers and clients will show security warnings. Accept the certificate or configure your system to trust it for development purposes.

#### Default Credentials

- Username: `DVC`
- Password: `uTAntEMTuVpcJucNjOJm`

**Important**: Change these credentials before production deployment.

#### Using MinIO Client (mc)

1. **Install MinIO Client**:
   ```bash
   # Download from https://docs.min.io/docs/minio-client-quickstart-guide.html
   ```

2. **Configure Alias**:
   ```bash
   mc alias set myminio https://localhost:30706 DVC uTAntEMTuVpcJucNjOJm
   ```

3. **Basic Operations**:
   ```bash
   mc mb myminio/my-bucket
   mc cp file.txt myminio/my-bucket/
   mc ls myminio/my-bucket
   ```

#### Using AWS CLI

1. **Configure AWS CLI**:
   ```bash
   aws configure
   # AWS Access Key ID: DVC
   # AWS Secret Access Key: uTAntEMTuVpcJucNjOJm
   # Default region name: us-east-1
   # Default output format: json
   ```

2. **Set MinIO Endpoint**:
   ```bash
   aws configure set default.s3.endpoint_url https://localhost:30706
   aws configure set default.s3.signature_version s3v4
   ```

3. **Basic Operations**:
   ```bash
   aws s3 mb s3://my-bucket
   aws s3 cp file.txt s3://my-bucket/
   aws s3 ls s3://my-bucket
   ```

## Security Considerations

### General Security

1. **Change Default Credentials**: Always modify `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` before production deployment. Use strong, unique passwords.

2. **Network Security**: 
   - Limit access to ports 30706 and 30707 to trusted networks.
   - Use firewalls to restrict access.
   - Consider using VPNs for remote access.

3. **Data Encryption**: 
   - MinIO supports server-side encryption. Enable it for sensitive data.
   - For the SSL setup, all data in transit is encrypted.

4. **Access Control**: 
   - Use MinIO's Identity and Access Management (IAM) features.
   - Create users, groups, and policies as needed.
   - Implement least-privilege access.

### SSL-Specific Security

1. **Certificate Management**:
   - Self-signed certificates are suitable for development but not production.
   - Obtain certificates from a trusted Certificate Authority (CA) for production.
   - Regularly renew certificates before expiration.

2. **Certificate Validation**:
   - Configure clients to validate certificates properly.
   - Avoid disabling certificate validation in production.

3. **Key Security**:
   - Protect private keys with appropriate file permissions.
   - Never commit certificates or keys to version control.

## Troubleshooting

### Common Issues

#### Port Conflicts
- **Symptom**: Docker Compose fails to start due to port binding errors.
- **Solution**: Check if ports 30706 or 30707 are in use by other services. Modify port mappings in `docker-compose.yaml` if necessary.

#### Volume Permission Issues
- **Symptom**: MinIO cannot write to the mounted volume.
- **Solution**: Ensure the host directory has appropriate read/write permissions for the Docker user (typically UID 1000).

#### SSL Certificate Errors
- **Symptom**: Connection refused or certificate validation errors.
- **Solution**: 
  - Verify certificates exist in the `certs` directory.
  - Check certificate validity and permissions.
  - For self-signed certificates, configure clients to accept them.

#### MinIO Not Starting
- **Symptom**: Container exits immediately.
- **Solution**: Check logs with `docker-compose logs dvc-minio`. Common causes include invalid configuration or insufficient resources.

### Logs and Debugging

- **View MinIO Logs**:
  ```bash
  docker-compose logs -f dvc-minio
  ```

- **View Certificate Generation Logs** (for SSL setup):
  ```bash
  cd openssl
  docker-compose logs
  ```

### Performance Tuning

- **Memory**: MinIO performs best with ample RAM. Monitor usage and adjust Docker memory limits if needed.
- **Disk I/O**: Use fast storage for the data volume to optimize performance.
- **Network**: Ensure sufficient bandwidth for your use case.

## Customization and Advanced Configuration

### Modifying Ports
Edit the `ports` section in `docker-compose.yaml` to change external port mappings.

### Changing Data Directory
Modify the `volumes` section to point to different host directories for data storage.

### Updating MinIO Version
Change the `image` tag in `docker-compose.yaml` to a different MinIO release. Check https://hub.docker.com/r/minio/minio/tags for available versions.

### Adding Nginx Reverse Proxy
For production deployments, consider adding an Nginx reverse proxy for load balancing, advanced SSL configuration, and additional security features.

### Environment-Specific Configurations
Create multiple `docker-compose.override.yaml` files for different environments (development, staging, production) with environment-specific settings.

## Contributing

This repository contains personal Docker configurations for MinIO. For MinIO-specific issues or contributions:

- Report MinIO bugs at https://github.com/minio/minio/issues
- Contribute to MinIO at https://github.com/minio/minio
- Check MinIO documentation at https://docs.min.io/

## License

- **MinIO**: Licensed under GNU AGPL v3.0
- **This Repository**: Configurations provided as-is without specific licensing. Use at your own risk.

## Additional Resources

- [MinIO Official Documentation](https://docs.min.io/)
- [MinIO GitHub Repository](https://github.com/minio/minio)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [DVC Documentation](https://dvc.org/doc)
- [AWS S3 API Compatibility](https://docs.min.io/docs/aws-s3-api)
- [MinIO Client Documentation](https://docs.min.io/docs/minio-client-quickstart-guide.html)
- [OpenSSL Documentation](https://www.openssl.org/docs/)

---

**Note**: This documentation is current as of October 2025. Check for updates to MinIO and Docker versions for the latest features and security patches.