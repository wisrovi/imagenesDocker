MinIO Docker Setups Documentation
==================================

.. image:: https://img.shields.io/badge/MinIO-Docker%20Setups-blue.svg
   :target: https://github.com/minio/minio
   :alt: MinIO

.. image:: https://img.shields.io/badge/Docker-Compose-green.svg
   :target: https://docs.docker.com/compose/
   :alt: Docker Compose

.. image:: https://img.shields.io/badge/License-AGPL--3.0-orange.svg
   :target: https://www.gnu.org/licenses/agpl-3.0.en.html
   :alt: License

Welcome to the comprehensive documentation for MinIO Docker Setups. This project provides Docker Compose configurations for deploying MinIO, a high-performance, S3-compatible object storage server, in two variants: a basic setup for development and an SSL-enabled setup for secure production environments.

Overview
--------

MinIO is a Kubernetes-native object storage server that is fully compatible with Amazon S3 APIs. This repository offers easy-to-deploy configurations using Docker, making it ideal for local development, testing, and integration with tools like Data Version Control (DVC) for machine learning workflows.

Key Features
~~~~~~~~~~~~

- **High Performance**: MinIO provides erasure coding, bitrot protection, and multi-cloud support for scalable storage.
- **S3 Compatibility**: Full API compatibility with Amazon S3, allowing seamless integration with existing tools.
- **Docker Deployment**: Simple containerized deployment using Docker Compose.
- **Two Configurations**: Basic setup for development and SSL-enabled setup for secure environments.
- **DVC Integration**: Optimized for use with DVC in machine learning pipelines.

Project Structure
~~~~~~~~~~~~~~~~~

.. code-block:: text

   MniIO/
   ├── MinIO-normal/
   │   ├── README.md
   │   └── docker-compose.yaml
   ├── Minio-ssl/
   │   ├── README.md
   │   ├── docker-compose.yaml
   │   ├── openssl/
   │   │   ├── docker-compose.yaml
   │   │   └── nginx/
   │   │       ├── Dockerfile
   │   │       └── conf/
   │   │           └── openssl_wisrovi.cnf
   │   └── docs/
   ├── docs/
   │   ├── conf.py
   │   ├── index.rst
   │   └── ...
   ├── README.md
   └── Makefile

Quick Start
~~~~~~~~~~~

For a quick start with the basic MinIO setup:

.. tabs::

   .. tab:: MinIO-Normal

      .. code-block:: bash

         cd MinIO-normal
         docker-compose up -d

      Access the console at http://localhost:30707

   .. tab:: MinIO-SSL

      .. code-block:: bash

         cd Minio-ssl
         cd openssl && docker-compose up  # Generate certificates
         cd .. && docker-compose up -d

      Access the console at https://localhost:30707

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   usage
   examples
   api_reference
   troubleshooting
   bibliography
   author

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`