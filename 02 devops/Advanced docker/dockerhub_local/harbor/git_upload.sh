#!/bin/bash
# Script to add, commit, and push all files individually to git

git add common.sh
git commit -m "Add common script with utility functions for Harbor installation"

git add docker-compose.trivy.yaml
git commit -m "Add Docker Compose file for Trivy adapter container"

git add docker-compose.yml
git commit -m "Add Docker Compose file for Harbor services including registry, database, and UI"

git add harbor.yml
git commit -m "Configure Harbor with HTTP, Trivy scanning, cache, metrics, trace, and user settings"

git add install.sh
git commit -m "Add Harbor installation script with Trivy enabled by default and sudo support"

git add prepare
git commit -m "Add prepare script to generate Harbor configurations from harbor.yml"

git add Readme.md
git commit -m "Add README with Harbor installation and usage instructions"

git add scripts/1.sh
git commit -m "Add script to pull Docker images from Docker Hub"

git add scripts/2.sh
git commit -m "Add script to tag Docker images for Harbor registry"

git add scripts/3.sh
git commit -m "Add script to push tagged images to Harbor registry"

git add scripts/4.sh
git commit -m "Add script to remove local Docker images after pushing"

git add scripts/5.sh
git commit -m "Add script to pull images from Harbor registry"

git add scripts/upload_multiple.sh
git commit -m "Add script to upload multiple Docker images using the individual scripts"

git add LICENSE
git commit -m "Add Apache License 2.0 for the project"

git add common/config/core/app.conf
git commit -m "Add Harbor core application configuration"

git add common/config/core/env
git commit -m "Add environment variables for Harbor core service"

git add common/config/db/env
git commit -m "Add environment variables for Harbor database service"

git add common/config/exporter/env
git commit -m "Add environment variables for Harbor exporter service"

git add common/config/jobservice/config.yml
git commit -m "Add configuration for Harbor job service"

git add common/config/jobservice/env
git commit -m "Add environment variables for Harbor job service"

git add common/config/log/logrotate.conf
git commit -m "Add logrotate configuration for Harbor logs"

git add common/config/log/rsyslog_docker.conf
git commit -m "Add rsyslog configuration for Docker logging in Harbor"

git add common/config/nginx/nginx.conf
git commit -m "Add Nginx configuration for Harbor proxy"

git add common/config/portal/nginx.conf
git commit -m "Add Nginx configuration for Harbor portal"

git add common/config/registry/config.yml
git commit -m "Add configuration for Harbor registry service"

git add common/config/registryctl/config.yml
git commit -m "Add configuration for Harbor registry controller"

git add common/config/registryctl/env
git commit -m "Add environment variables for Harbor registry controller"

git add common/config/registry/passwd
git commit -m "Add password file for Harbor registry authentication"

git add common/config/registry/root.crt
git commit -m "Add root certificate for Harbor registry"

git add harbor-online-installer-v2.14.0.tgz
git commit -m "Add Harbor online installer tarball for version 2.14.0"

git push