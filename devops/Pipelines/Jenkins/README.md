# Jenkins CI/CD Pipeline Setup for Cybersecurity Projects

## Overview

This project provides a Dockerized Jenkins environment specifically configured for continuous integration and continuous deployment (CI/CD) pipelines in cybersecurity-related projects. The setup includes Jenkins with Docker support, Git integration, and Python 3.10 for scripting and automation tasks commonly used in security assessments and tooling.

The configuration allows for secure, isolated pipeline execution while maintaining access to host Docker and Git resources, making it ideal for automated security testing, vulnerability scanning, and deployment workflows.

## Features

- **Docker Integration**: Full Docker support for containerized builds and deployments
- **Git Support**: SSH key integration for private repository access
- **Python Environment**: Python 3.10 installed for security scripting and automation
- **Persistent Configuration**: Jenkins home directory mounted for configuration persistence
- **Privileged Mode**: Access to host Docker socket for advanced container operations
- **Recommended Plugins**: Pre-configured with essential plugins for CI/CD and security workflows

## Prerequisites

Before setting up this Jenkins environment, ensure you have the following installed on your host system:

- Docker Engine (version 20.10 or later)
- Docker Compose (version 1.29 or later)
- Git (for repository operations)
- SSH keys configured for Git access (optional, but recommended for private repos)

## Installation and Setup

1. **Clone or Download the Project**:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Build and Start Jenkins**:
   ```bash
   docker-compose up -d --build
   ```

   This command will:
   - Build the custom Jenkins image with additional tools
   - Start the Jenkins container
   - Mount necessary volumes for persistence and access

3. **Access Jenkins**:
   - Open your browser and navigate to `http://localhost:50443`
   - Follow the initial setup wizard to configure Jenkins
   - Install recommended plugins when prompted

## File Structure

- **`Dockerfile`**: This file defines the custom Jenkins image. It starts from the official Jenkins Alpine image and adds Docker, Git, Docker Compose, and Python 3.10. This setup provides a complete environment for building, testing, and deploying applications, especially those requiring Python scripts for automation or security tasks.

- **`docker-compose.yaml`**: This file orchestrates the deployment of the Jenkins environment. It builds the custom Docker image, configures port mappings, and mounts volumes for data persistence and host system interaction. The key configurations include:
    - **`jenkins_home`**: A volume to persist Jenkins data, ensuring that jobs, plugins, and configurations are not lost when the container restarts.
    - **Docker Socket**: Mounting `/var/run/docker.sock` allows the Jenkins container to control the host's Docker daemon, enabling Docker-in-Docker workflows.
    - **SSH Keys**: The `~/.ssh` directory is mounted to allow Jenkins to authenticate with Git repositories using your local SSH keys.

- **`structure.txt`**: This file outlines the recommended directory structure for organizing Jenkins pipeline files within your projects. Following this structure helps maintain consistency and clarity in your CI/CD workflows.

## Configuration

### Environment Variables

- `TZ=Europe/Madrid`: Sets the timezone for the Jenkins container. Modify this in `docker-compose.yaml` to match your location.

### Volumes

The `docker-compose.yaml` file configures the following volume mounts:

- `/home/wisrovi/projects:/home/projects`: Shared directory for project files and Git repositories
- `./jenkins_home:/var/jenkins_home`: Persistent storage for Jenkins configuration and data
- `/var/run/docker.sock:/var/run/docker.sock`: Access to host Docker daemon
- `~/.ssh:/root/.ssh`: SSH keys for Git authentication

### Ports

- `50443:8080`: Jenkins web interface (mapped to port 50443 on host)
- `50000:50000`: Jenkins agent port (commented out by default)

## Recommended Plugins

For optimal functionality in cybersecurity CI/CD pipelines, install the following plugins:

- **Role-based Authorization Strategy**: Implements role-based access control for secure pipeline management
- **Docker Pipeline Plugin**: Enables Docker commands within Jenkins pipelines
- **Blue Ocean Plugin**: Provides a modern, user-friendly interface for pipeline visualization
- **GitHub Plugin**: Triggers pipelines on GitHub events (merges, pushes, etc.)
- **SonarQube Scanner Plugin**: Integrates automated code quality and security scanning

## Usage

### Creating Pipelines

1. Log into Jenkins via the web interface
2. Create a new pipeline job
3. Configure the pipeline to use your Git repository
4. Define your CI/CD stages (build, test, deploy, security scan, etc.)

### Example Pipeline Stage

```groovy
pipeline {
    agent any
    stages {
        stage('Security Scan') {
            steps {
                sh 'python3.10 -m pip install -r requirements.txt'
                sh 'python3.10 security_scanner.py'
            }
        }
        stage('Build') {
            steps {
                sh 'docker build -t my-security-tool .'
            }
        }
    }
}
```

### Managing SSH Keys

Ensure your SSH keys are properly configured in `~/.ssh` on the host system. The container mounts this directory, allowing Jenkins to authenticate with Git repositories.

## Project Structure for Jenkins Pipelines

To maintain organized and scalable CI/CD workflows, each project integrated with this Jenkins setup should follow a standardized directory structure for pipeline definitions.

### Required Structure

Every project repository should include a `jenkins` folder at the root level. This folder serves as the central location for all Jenkins-related configurations and pipeline scripts.

#### Folder Organization

The `jenkins` folder should contain a subfolder named after the project (e.g., `<project>`), which organizes pipelines by type and environment.

#### Pipeline Files

Pipeline definitions are stored as `.jenkinsfile` files, with specific naming conventions for environments: `001-DEVELOPMENT.jenkinsfile`, `002-TEST.jenkinsfile`, `003-PRODUCTION.jenkinsfile`.

Example structure based on the provided template:
```
jenkins/
    |
    |------ <project>
    |                 |
    |                 |---------- build
    |                                      |
    |                                      |----------- 1-environment_preparation
    |                                                                           |----------- 001-DEVELOPMENT.jenkinsfile
    |                                                                           |----------- 002-TEST.jenkinsfile
    |                                                                           |----------- 003-PRODUCTION.jenkinsfile
    |                                      |----------- 2-db_config.jenkinsfile
    |                 |---------- deploy   // make stop ; git pull ; make build ; make start
    |                                      |----------- 001-DEVELOPMENT.jenkinsfile
    |                                      |----------- 002-TEST.jenkinsfile
    |                                      |----------- 003-PRODUCTION.jenkinsfile
    |                 |---------- test    // pytest
    |                                      |----------- 001-DEVELOPMENT.jenkinsfile
    |                                      |----------- 002-TEST.jenkinsfile
    |                                      |----------- 003-PRODUCTION.jenkinsfile
    |                 |---------- QA    // sonarqube - flake8 - pep8 - pylint
    |                                      |----------- 001-DEVELOPMENT.jenkinsfile
    |                                      |----------- 002-TEST.jenkinsfile
    |                                      |----------- 003-PRODUCTION.jenkinsfile
    |                 |---------- PR reviewer
    |                                      |----------- 001-DEVELOPMENT.jenkinsfile
    |                                      |----------- 002-TEST.jenkinsfile
    |                                      |----------- 003-PRODUCTION.jenkinsfile
    |                 |---------- train
    |                                      |----------- 0-data_preparation.jenkinsfile     // download data of DVC, convert to model format, data incrementation, etc
    |                                      |----------- 1-yolo_model.jenkinsfile     // send to train_service (only for yolo), so that 3D uses the respective pipeline
    |                                      |----------- 2-download_best_results.jenkinsfile     // mlflow download the results of experiments trained
    |                                      |----------- 3-compare_with_last.jenkinsfile     // select better experiment trained and compare with the actual model in the github
    |                                      |----------- 4-statistics.jenkinsfile     // to find the metrics and statistics
    |                                      |----------- 5-deploy_best_model.jenkinsfile     // replace the last model with the new model
```

#### Best Practices

- **Modular Pipelines**: Break down complex workflows into smaller, reusable pipeline files organized by stage and environment
- **Environment-Specific Files**: Use numbered prefixes (001, 002, 003) for DEVELOPMENT, TEST, and PRODUCTION environments respectively
- **Naming Conventions**: Follow the exact naming pattern shown, with descriptive folder names and comments indicating the purpose
- **Documentation**: Include inline comments in pipeline files explaining each stage and step, as shown in the structure
- **Version Control**: Keep pipeline files under version control alongside the project code

This structure ensures that pipeline configurations are maintainable, versioned, and easily understandable by team members working on cybersecurity and machine learning projects.

## Security Considerations

- **Privileged Mode**: The container runs in privileged mode to access the Docker socket. Use with caution in production environments.
- **Volume Permissions**: Ensure proper file permissions on mounted volumes to prevent unauthorized access.
- **Network Security**: Configure firewall rules to restrict access to Jenkins ports.
- **Plugin Security**: Regularly update plugins and monitor for security vulnerabilities.
- **Secret Management**: Use Jenkins credentials store for sensitive information like API keys and passwords.

## Troubleshooting

### Common Issues

1. **Permission Denied on Docker Socket**:
   - Ensure the user running Docker Compose has access to `/var/run/docker.sock`
   - Check file permissions: `ls -la /var/run/docker.sock`

2. **SSH Key Issues**:
   - Verify SSH keys are in `~/.ssh` and have correct permissions (600)
   - Test SSH connection manually: `ssh -T git@github.com`

3. **Port Conflicts**:
   - If port 50443 is in use, modify the port mapping in `docker-compose.yaml`

4. **Build Failures**:
   - Check Docker logs: `docker-compose logs jenkins`
   - Ensure all required dependencies are installed in the Dockerfile

### Logs

View Jenkins logs with:
```bash
docker-compose logs -f jenkins
```

## Contributing

Contributions to improve this Jenkins setup are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request with a clear description of changes

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Resources

- [Jenkins Official Documentation](https://www.jenkins.io/doc/)
- [Docker Pipeline Plugin](https://plugins.jenkins.io/docker-workflow/)
- [Blue Ocean Plugin](https://plugins.jenkins.io/blueocean/)
- [SonarQube Integration](https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/jenkins-extension-sonarqube/)

For cybersecurity-specific CI/CD best practices, refer to OWASP and NIST guidelines.
