# MinIO SSL Setup

This project provides a complete Docker-based setup for deploying MinIO, an S3-compatible object storage server, with SSL/TLS encryption enabled. It includes automated certificate generation using OpenSSL and configuration for secure access.

## Overview

MinIO is a high-performance, distributed object storage system that is API-compatible with Amazon S3. This setup ensures secure communication through SSL/TLS certificates, making it suitable for production environments or secure development setups.

The project consists of:
- MinIO server with SSL configuration
- Automated SSL certificate generation using OpenSSL
- Docker Compose configurations for easy deployment

## Prerequisites

- Docker and Docker Compose installed on your system
- Basic understanding of Docker and containerization
- Access to ports 30706 (MinIO S3 API) and 30707 (MinIO Console) on your host machine

## Project Structure

```
Minio-ssl/
├── docker-compose.yaml          # Main MinIO service configuration
├── openssl/
│   ├── docker-compose.yaml      # Certificate generation service
│   ├── nginx/
│   │   ├── Dockerfile           # OpenSSL container build file
│   │   └── conf/
│   │       └── openssl_wisrovi.cnf  # OpenSSL configuration for certificate
│   └── Readme.md                # Legacy documentation (can be ignored)
└── Readme.md                    # This file
```

## Installation and Setup

### 1. Clone or Download the Project

Ensure you have the project files in your desired directory.

### 2. Generate SSL Certificates

First, generate the SSL certificates required for MinIO:

```bash
cd openssl
docker-compose up
```

This command will:
- Build a Docker image with OpenSSL
- Generate a self-signed SSL certificate valid for 825 days
- Place the certificates in the `certs` directory at the project root

The generated files will be:
- `certs/fullchain.pem`: The public certificate
- `certs/privkey.pem`: The private key

### 3. Start MinIO Server

Once certificates are generated, start the MinIO server:

```bash
docker-compose up -d
```

This will start MinIO with SSL enabled on the configured ports.

## Configuration

### MinIO Configuration

The main `docker-compose.yaml` file contains the following key configurations:

- **Image**: `minio/minio:RELEASE.2025-02-28T09-55-16Z` (latest stable release)
- **Ports**:
  - `30706`: MinIO S3 API (mapped to container port 9000)
  - `30707`: MinIO Web Console (mapped to container port 9001)
- **Environment Variables**:
  - `MINIO_ROOT_USER`: Admin username (default: DVC)
  - `MINIO_ROOT_PASSWORD`: Admin password (default: uTAntEMTuVpcJucNjOJm)
  - `MINIO_CERT_PUBLIC_KEY`: Path to public certificate
  - `MINIO_CERT_PRIVATE_KEY`: Path to private key
- **Volumes**:
  - `./DVC_data:/data`: Persistent storage for MinIO data
  - `./certs:/root/.minio/certs`: SSL certificates

### SSL Certificate Configuration

The SSL certificates are configured in `openssl/nginx/conf/openssl_wisrovi.cnf`:

- **Domain**: www.dvc.ecapturedtech.com
- **Validity**: 825 days
- **Key Size**: 4096-bit RSA
- **Subject Alternative Names**: Includes https://ecapturedtech.com/

## Usage

### Accessing MinIO

1. **Web Console**: Open https://localhost:30707 in your browser
2. **S3 API**: Use any S3-compatible client with endpoint https://localhost:30706

### Default Credentials

- **Username**: DVC
- **Password**: uTAntEMTuVpcJucNjOJm

**Important**: Change these default credentials in production environments!

### Basic Operations

#### Using MinIO Client (mc)

Install MinIO Client and configure:

```bash
mc alias set myminio https://localhost:30706 DVC uTAntEMTuVpcJucNjOJm
```

Create a bucket:
```bash
mc mb myminio/my-bucket
```

Upload a file:
```bash
mc cp myfile.txt myminio/my-bucket/
```

#### Using AWS CLI

Configure AWS CLI for MinIO:

```bash
aws configure
# AWS Access Key ID: DVC
# AWS Secret Access Key: uTAntEMTuVpcJucNjOJm
# Default region name: us-east-1
# Default output format: json
```

Set endpoint:
```bash
aws configure set default.s3.endpoint_url https://localhost:30706
aws configure set default.s3.signature_version s3v4
```

Create a bucket:
```bash
aws s3 mb s3://my-bucket
```

## Security Considerations

1. **Certificate Validity**: The generated certificates are self-signed. For production, use certificates from a trusted Certificate Authority (CA).

2. **Default Credentials**: Change the default MINIO_ROOT_USER and MINIO_ROOT_PASSWORD before deploying to production.

3. **Network Security**: Ensure the exposed ports (30706, 30707) are properly secured in your network configuration.

4. **Data Persistence**: The `./DVC_data` directory contains all MinIO data. Back up this directory regularly.

## Troubleshooting

### Certificate Issues

If you encounter SSL certificate errors:
1. Ensure certificates were generated successfully in the `certs` directory
2. Check that `fullchain.pem` and `privkey.pem` exist and have correct permissions
3. For browser access, you may need to accept the self-signed certificate

### Port Conflicts

If ports 30706 or 30707 are already in use:
1. Stop other services using these ports
2. Modify the port mappings in `docker-compose.yaml`

### Permission Issues

Ensure Docker has access to the project directory and can create the `certs` and `DVC_data` directories.

## Customization

### Changing Domain/Certificate Details

Edit `openssl/nginx/conf/openssl_wisrovi.cnf` to modify:
- Domain name (CN and subjectAltName)
- Organization details
- Certificate validity period

### Modifying MinIO Configuration

Edit `docker-compose.yaml` to change:
- Ports
- Credentials
- Data directory location
- MinIO version

### Adding Nginx Reverse Proxy

For production deployments, consider adding an Nginx reverse proxy in front of MinIO for additional features like load balancing and advanced SSL configuration.

## Contributing

This is a personal project setup. For improvements or issues, please refer to the MinIO documentation at https://docs.min.io/

## License

This project setup is provided as-is without any specific license. MinIO itself is licensed under the GNU AGPL v3.

## Resources

- [MinIO Documentation](https://docs.min.io/)
- [MinIO GitHub Repository](https://github.com/minio/minio)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [OpenSSL Documentation](https://www.openssl.org/docs/)