# MinIO Docker Setup for DVC

This project provides a Docker Compose configuration to deploy a MinIO server, an S3-compatible object storage server, optimized for use with Data Version Control (DVC). MinIO is used here to store and manage datasets and models in a distributed, scalable manner.

## Overview

MinIO is a high-performance, Kubernetes-native object storage server that is fully compatible with Amazon S3 APIs. This setup allows you to run MinIO locally or in a containerized environment, making it ideal for development, testing, and integration with tools like DVC for machine learning workflows.

The configuration includes:
- MinIO server running on a specific version
- Pre-configured root user and password
- Mapped ports for S3 API and web console access
- Persistent volume mounting for data storage

## Prerequisites

Before running this setup, ensure you have the following installed on your system:

- Docker (version 20.10 or later recommended)
- Docker Compose (version 2.0 or later)
- Sufficient disk space for your data storage needs (mounted at `/mnt/DVC_tmp/DVC_data` or equivalent)

## Installation and Setup

1. **Clone or Download the Project**:
   Ensure you have this `docker-compose.yaml` file in your project directory.

2. **Configure Volumes**:
   - The default volume mount is set to `/mnt/DVC_tmp/DVC_data:/data`.
   - If this path does not exist on your host system, create it or modify the volume mapping in `docker-compose.yaml` to point to an appropriate directory.
   - Example: Change `- /mnt/DVC_tmp/DVC_data:/data` to `- ./DVC_data:/data` for a local directory.

3. **Environment Variables**:
   - `MINIO_ROOT_USER`: Set to `DVC` (default admin username)
   - `MINIO_ROOT_PASSWORD`: Set to `uTAntEMTuVpcJucNjOJm` (change this for production use)
   - Optional: Uncomment `MINIO_DEFAULT_BUCKETS=datasets` to create a default bucket named "datasets" on startup.

## Usage

### Starting the MinIO Server

To start the MinIO server, run the following command in the directory containing `docker-compose.yaml`:

```bash
docker-compose up -d
```

This will:
- Pull the MinIO image if not already present
- Start the MinIO server in detached mode
- Map the necessary ports
- Mount the specified volume for data persistence

### Accessing MinIO

- **S3 API Endpoint**: `http://localhost:30706`
- **Web Console (Admin UI)**: `http://localhost:30707`

Log in to the web console using:
- Username: `DVC`
- Password: `uTAntEMTuVpcJucNjOJm`

### Integration with DVC

This MinIO setup is designed for use with DVC. To configure DVC to use this MinIO instance:

1. Install DVC if not already installed:
   ```bash
   pip install dvc
   ```

2. Configure DVC remote storage:
   ```bash
   dvc remote add -d myremote s3://bucket-name
   dvc remote modify myremote endpointurl http://localhost:30706
   dvc remote modify myremote access_key_id DVC
   dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm
   ```

3. Create a bucket in MinIO (via web console or CLI) and use it in your DVC workflows.

### Stopping the Server

To stop the MinIO server:

```bash
docker-compose down
```

This will stop and remove the containers, but your data will persist in the mounted volume.

## Configuration Details

### Ports
- `30706:9000` - S3 API port (external:30706, internal:9000)
- `30707:9001` - MinIO Console port (external:30707, internal:9001)

### Volumes
- `/mnt/DVC_tmp/DVC_data:/data` - Mounts host directory to container's `/data` for persistent storage

### Environment Variables
- `MINIO_ROOT_USER`: Root username for MinIO
- `MINIO_ROOT_PASSWORD`: Root password for MinIO
- `MINIO_DEFAULT_BUCKETS`: (Commented out) Default buckets to create on startup

## Security Considerations

- **Change Default Credentials**: The default password is provided for convenience. For production deployments, generate strong, unique credentials.
- **Network Security**: Ensure ports 30706 and 30707 are not exposed publicly unless necessary.
- **Data Encryption**: MinIO supports server-side encryption. Consider enabling it for sensitive data.
- **Access Control**: Use MinIO's IAM features to manage users, groups, and policies.

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If ports 30706 or 30707 are already in use, modify the port mappings in `docker-compose.yaml`.

2. **Volume Permissions**: Ensure the host directory has appropriate read/write permissions for the Docker user.

3. **MinIO Not Starting**: Check Docker logs with `docker-compose logs dvc-minio` for error messages.

4. **Connection Refused**: Verify that MinIO is running and accessible on the specified ports.

### Logs

View MinIO logs:
```bash
docker-compose logs -f dvc-minio
```

## Contributing

If you encounter issues or have suggestions for improvements, please create an issue in the project repository.

## License

This configuration is provided as-is. MinIO itself is licensed under the Apache License 2.0.

## Additional Resources

- [MinIO Documentation](https://docs.min.io/)
- [DVC Documentation](https://dvc.org/doc)
- [Docker Compose Documentation](https://docs.docker.com/compose/)