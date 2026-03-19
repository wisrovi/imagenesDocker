#!/bin/bash

for file in $(git ls-files); do
  # determine prefix
  if [[ $file == *.rst ]] || [[ $file == *.md ]] || [[ $file == *.html ]] || [[ $file == *.txt ]] || [[ $file == *.css ]] || [[ $file == *.js ]] || [[ $file == *.doctree ]] || [[ $file == *.pickle ]] || [[ $file == *.inv ]] || [[ $file == *.svg ]] || [[ $file == *.png ]] || [[ $file == *buildinfo ]]; then
    prefix="[DOC]"
  elif [[ $file == Makefile ]] || [[ $file == make.bat ]] || [[ $file == conf.py ]]; then
    prefix="[REFACTOR]"
  else
    prefix="[FEATURE]"
  fi

  # description
  if [[ $file == README.md ]]; then
    description="This file provides an overview and detailed instructions for deploying MinIO using Docker Compose in two configurations: normal and SSL-enabled."
  elif [[ $file == Makefile ]]; then
    description="This file contains a set of make targets for building documentation, starting and stopping MinIO services, generating SSL certificates, testing, and other utility functions for the MinIO Docker project."
  elif [[ $file == mapper_files.sh ]]; then
    description="This bash script automates the process of adding and committing all tracked files in the repository with appropriate prefixes and descriptions based on their content."
  elif [[ $file == docs/conf.py ]]; then
    description="This file configures Sphinx for building the documentation, including project information, extensions, and HTML output settings."
  elif [[ $file == docs/Makefile ]]; then
    description="This Makefile provides targets for building and cleaning the Sphinx documentation."
  elif [[ $file == docs/index.rst ]]; then
    description="This reStructuredText file serves as the main page for the Sphinx documentation, providing an overview of the MinIO Docker Setups project, its features, and structure."
  elif [[ $file == docs/api_reference.rst ]]; then
    description="This reStructuredText file contains the API reference documentation for the MinIO Docker setups."
  elif [[ $file == docs/author.rst ]]; then
    description="This reStructuredText file provides information about the author of the MinIO Docker setups project."
  elif [[ $file == docs/bibliography.rst ]]; then
    description="This reStructuredText file contains the bibliography and references for the project documentation."
  elif [[ $file == docs/examples.rst ]]; then
    description="This reStructuredText file provides examples of using the MinIO Docker setups."
  elif [[ $file == docs/installation.rst ]]; then
    description="This reStructuredText file details the installation process for the MinIO Docker setups."
  elif [[ $file == docs/overview.rst ]]; then
    description="This reStructuredText file gives an overview of the MinIO Docker setups and their features."
  elif [[ $file == docs/troubleshooting.rst ]]; then
    description="This reStructuredText file provides troubleshooting guides for common issues with the MinIO Docker setups."
  elif [[ $file == docs/usage.rst ]]; then
    description="This reStructuredText file explains how to use the MinIO Docker setups."
  elif [[ $file == MinIO-normal/docker-compose.yaml ]]; then
    description="This Docker Compose file defines two services: dvc-minio for running the MinIO server with persistent storage and default credentials, and docs for serving the built documentation via Nginx."
  elif [[ $file == MinIO-normal/README.md ]]; then
    description="This file provides detailed documentation for deploying and using the MinIO-Normal setup, including installation, configuration, usage with DVC, and troubleshooting."
  elif [[ $file == MinIO-normal/Makefile ]]; then
    description="This Makefile provides build targets for the MinIO-Normal setup, including commands to start, stop, and manage the Docker services."
  elif [[ $file == MinIO-normal/docs/Makefile ]]; then
    description="This Makefile provides targets for building and cleaning the documentation for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/api_reference.rst ]]; then
    description="This reStructuredText file contains the API reference for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/bibliography.rst ]]; then
    description="This reStructuredText file contains references for the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/configuration.rst ]]; then
    description="This reStructuredText file details the configuration options for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/examples.rst ]]; then
    description="This reStructuredText file provides examples for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/index.rst ]]; then
    description="This reStructuredText file is the main documentation page for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/installation.rst ]]; then
    description="This reStructuredText file details the installation for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/troubleshooting.rst ]]; then
    description="This reStructuredText file provides troubleshooting for the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/usage.rst ]]; then
    description="This reStructuredText file explains usage of the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/conf.py ]]; then
    description="This file configures Sphinx for the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/make.bat ]]; then
    description="This batch file provides build commands for Windows users for the MinIO-Normal documentation."
  elif [[ $file == Minio-ssl/docker-compose.yaml ]]; then
    description="This Docker Compose file defines the MinIO server with SSL certificates mounted, environment variables for secure access, and persistent storage."
  elif [[ $file == Minio-ssl/Readme.md ]]; then
    description="This file provides detailed documentation for deploying and using the MinIO-SSL setup, including SSL certificate generation, secure configuration, and usage instructions."
  elif [[ $file == Minio-ssl/Makefile ]]; then
    description="This Makefile provides build targets for the MinIO-SSL setup, including commands to generate certificates, start, stop, and manage the secure Docker services."
  elif [[ $file == Minio-ssl/docs/Makefile ]]; then
    description="This Makefile provides targets for building and cleaning the documentation for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/api-reference.rst ]]; then
    description="This reStructuredText file contains the API reference for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/author.rst ]]; then
    description="This reStructuredText file provides author information for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/bibliography.rst ]]; then
    description="This reStructuredText file contains references for the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/configuration.rst ]]; then
    description="This reStructuredText file details the configuration for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/examples.rst ]]; then
    description="This reStructuredText file provides examples for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/index.rst ]]; then
    description="This reStructuredText file is the main documentation page for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/installation.rst ]]; then
    description="This reStructuredText file details the installation for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/overview.rst ]]; then
    description="This reStructuredText file gives an overview of the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/troubleshooting.rst ]]; then
    description="This reStructuredText file provides troubleshooting for the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/usage.rst ]]; then
    description="This reStructuredText file explains usage of the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/conf.py ]]; then
    description="This file configures Sphinx for the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_static/custom.css ]]; then
    description="This CSS file provides custom styling for the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/openssl/docker-compose.yaml ]]; then
    description="This Docker Compose file defines a service to run OpenSSL in a container for generating SSL certificates."
  elif [[ $file == Minio-ssl/openssl/Readme.md ]]; then
    description="This file contains legacy documentation for the OpenSSL certificate generation process."
  elif [[ $file == Minio-ssl/openssl/nginx/Dockerfile ]]; then
    description="This Dockerfile creates an image with OpenSSL installed for certificate generation."
  elif [[ $file == Minio-ssl/openssl/nginx/conf/openssl_wisrovi.cnf ]]; then
    description="This OpenSSL configuration file specifies the certificate details, including domain names and validity period."
  elif [[ $file =~ ^docs/_build/html/.*\.html$ ]]; then
    page=$(basename "$file" .html)
    description="This HTML file is the rendered $page page of the Sphinx documentation."
  elif [[ $file =~ ^docs/_build/html/_static/.*\.css$ ]]; then
    description="This CSS file provides styling for the Sphinx documentation."
  elif [[ $file =~ ^docs/_build/html/_static/.*\.js$ ]]; then
    description="This JavaScript file provides functionality for the Sphinx documentation."
  elif [[ $file =~ ^docs/_build/html/_static/.*\.(svg|png)$ ]]; then
    description="This image file is used in the Sphinx documentation."
  elif [[ $file =~ ^docs/_build/html/_static/fonts/ ]]; then
    description="This font file is used for styling the Sphinx documentation."
  elif [[ $file == docs/_build/html/.buildinfo ]]; then
    description="This file contains build information for the Sphinx documentation."
  elif [[ $file == docs/_build/html/objects.inv ]]; then
    description="This file is the intersphinx inventory for cross-references in the documentation."
  elif [[ $file == docs/_build/html/searchindex.js ]]; then
    description="This JavaScript file contains the search index for the Sphinx documentation."
  elif [[ $file == docs/_build/html/genindex.html ]]; then
    description="This HTML file is the general index of the Sphinx documentation."
  elif [[ $file == docs/_build/html/search.html ]]; then
    description="This HTML file provides the search interface for the Sphinx documentation."
  elif [[ $file =~ ^docs/_build/doctrees/.*\.doctree$ ]]; then
    description="This doctree file contains the parsed structure for a documentation page in Sphinx."
  elif [[ $file == docs/_build/environment.pickle ]]; then
    description="This pickle file stores the build environment data for Sphinx."
  elif [[ $file =~ ^docs/_build/html/_sources/.*\.txt$ ]]; then
    description="This text file is the source copy of a reStructuredText file for the documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/.*\.html$ ]]; then
    page=$(basename "$file" .html)
    description="This HTML file is the rendered $page page of the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/_static/.*\.css$ ]]; then
    description="This CSS file provides styling for the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/_static/.*\.js$ ]]; then
    description="This JavaScript file provides functionality for the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/_static/.*\.(svg|png)$ ]]; then
    description="This image file is used in the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/_static/fonts/ ]]; then
    description="This font file is used for styling the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/_build/html/.buildinfo ]]; then
    description="This file contains build information for the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/_build/html/objects.inv ]]; then
    description="This file is the intersphinx inventory for cross-references in the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/_build/html/searchindex.js ]]; then
    description="This JavaScript file contains the search index for the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/_build/html/genindex.html ]]; then
    description="This HTML file is the general index of the MinIO-Normal documentation."
  elif [[ $file == MinIO-normal/docs/_build/html/search.html ]]; then
    description="This HTML file provides the search interface for the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/doctrees/.*\.doctree$ ]]; then
    description="This doctree file contains the parsed structure for a documentation page in the MinIO-Normal setup."
  elif [[ $file == MinIO-normal/docs/_build/environment.pickle ]]; then
    description="This pickle file stores the build environment data for the MinIO-Normal documentation."
  elif [[ $file =~ ^MinIO-normal/docs/_build/html/_sources/.*\.txt$ ]]; then
    description="This text file is the source copy of a reStructuredText file for the MinIO-Normal documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/.*\.html$ ]]; then
    page=$(basename "$file" .html)
    description="This HTML file is the rendered $page page of the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/_static/.*\.css$ ]]; then
    description="This CSS file provides styling for the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/_static/.*\.js$ ]]; then
    description="This JavaScript file provides functionality for the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/_static/.*\.(svg|png)$ ]]; then
    description="This image file is used in the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/_static/fonts/ ]]; then
    description="This font file is used for styling the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_build/html/.buildinfo ]]; then
    description="This file contains build information for the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_build/html/objects.inv ]]; then
    description="This file is the intersphinx inventory for cross-references in the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_build/html/searchindex.js ]]; then
    description="This JavaScript file contains the search index for the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_build/html/genindex.html ]]; then
    description="This HTML file is the general index of the MinIO-SSL documentation."
  elif [[ $file == Minio-ssl/docs/_build/html/search.html ]]; then
    description="This HTML file provides the search interface for the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/doctrees/.*\.doctree$ ]]; then
    description="This doctree file contains the parsed structure for a documentation page in the MinIO-SSL setup."
  elif [[ $file == Minio-ssl/docs/_build/environment.pickle ]]; then
    description="This pickle file stores the build environment data for the MinIO-SSL documentation."
  elif [[ $file =~ ^Minio-ssl/docs/_build/html/_sources/.*\.txt$ ]]; then
    description="This text file is the source copy of a reStructuredText file for the MinIO-SSL documentation."
  else
    description="This file is part of the MinIO Docker project."
  fi

  git add "$file"
  git commit -m "[$prefix] $description"
done