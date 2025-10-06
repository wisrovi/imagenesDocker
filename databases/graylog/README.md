# Graylog Docker Setup with Python Log Sender

This project provides a complete Docker-based setup for deploying Graylog, a powerful log management platform, along with its dependencies (MongoDB and OpenSearch). It includes Python scripts for sending test logs to Graylog via GELF (Graylog Extended Log Format) over UDP.

## Overview

Graylog is an open-source log aggregation, indexing, and visualization tool that allows you to collect, process, and analyze log data from various sources. This setup uses:

- **Graylog 5.2**: The main log management server.
- **MongoDB 6.0**: Stores Graylog's configuration and metadata.
- **OpenSearch 2.12.0**: Provides search and analytics capabilities for log data.

The included Python scripts (`send_logs.py` and `send_logs_loguru.py`) demonstrate how to send structured logs to Graylog, including custom fields for filtering and categorization.

## Features

- **Complete Docker Setup**: Easy deployment with Docker Compose.
- **Secure Configuration**: Includes proper authentication and security settings.
- **Python Log Integration**: Scripts for sending logs with different logging libraries.
- **Customizable**: Environment variables for configuration.
- **Persistent Data**: Volumes for data persistence across container restarts.

## Prerequisites

Before running this setup, ensure you have the following installed:

- **Docker**: Version 20.10 or later.
- **Docker Compose**: Version 2.0 or later.
- **Python 3.8+**: For running the log sender script.

## Installation

1. **Clone or Download the Repository**:
   ```bash
   git clone <repository-url>
   cd graylog-docker-setup
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Services**:
   ```bash
   docker-compose up -d
   ```

   This will start MongoDB, OpenSearch, and Graylog containers. The initial startup may take a few minutes.

4. **Verify Services**:
   ```bash
   docker-compose ps
   ```

   All services should show as "Up".

## Configuration

### Environment Variables

The setup uses environment variables for configuration. You can override defaults by creating a `.env` file or setting them in your shell.

#### Docker Compose Variables

- `GRAYLOG_ROOT_PASSWORD_SHA2`: SHA2 hash of the admin password (default: hash for "admin").
- `GRAYLOG_HTTP_EXTERNAL_URI`: External URI for Graylog web interface (default: `http://127.0.0.1:9100/`).
- `OPENSEARCH_INITIAL_ADMIN_PASSWORD`: Password for OpenSearch admin (default: `TranquilSunset42BlueOcean!`).

### Ports

- **Graylog Web UI**: `http://localhost:9100`
- **Graylog UDP Input**: `12201/udp`
- **OpenSearch REST API**: `http://localhost:9200`

### Volumes

- `mongo_data`: Persists MongoDB data.
- `opensearch_data`: Persists OpenSearch data.
- `graylog_data`: Persists Graylog data.

## Usage

### Accessing Graylog

1. Open your browser and navigate to `http://localhost:9100`.
2. Log in with username `admin` and password `admin`.
3. Create a GELF UDP input on port 12201 in the `System -> Inputs` section of the Graylog web interface.

### Running the Python Log Scripts

To send test logs to your Graylog instance, run one of the following commands:

```bash
# Basic script using the standard logging library
python send_logs.py

# Modern script using the loguru library
python send_logs_loguru.py
```

These will send test log messages to the GELF UDP input you configured in Graylog.

### Filtering Logs

In Graylog, you can use the search bar to filter logs by the custom fields:

- `microservice:queso`
- `version:pan`
- `tag:integrated_tag` (for `send_logs_loguru.py`)

## Troubleshooting

### Common Issues

1. **Graylog Not Starting**:
   - Check the logs for errors: `docker-compose logs graylog`
   - Ensure that the `GRAYLOG_ROOT_PASSWORD_SHA2` is a valid SHA2 hash.

2. **Cannot Connect to Graylog**:
   - Verify that the GELF UDP input is configured correctly in the Graylog web interface.
   - Check your firewall settings to ensure that UDP traffic on port 12201 is allowed.

### Resetting Data

To reset all data and start with a clean environment, run the following commands:

```bash
docker-compose down -v
docker-compose up -d
```

## Scripts

### send_logs.py

A basic script using the standard `logging` library with a GELF handler to send logs to Graylog.

### send_logs_loguru.py

A simplified and modern script using the `loguru` library for structured logging to Graylog.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test thoroughly.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.