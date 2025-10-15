# Jenkins Pipelines for Train Service

## Overview

This repository provides a complete, containerized environment for running Jenkins pipelines. It includes a sample pipeline designed for a machine learning "Train Service," demonstrating how to automate tasks within a Jenkins CI/CD workflow. The entire setup is managed via Docker and Docker Compose, making it portable and easy to deploy.

This project also includes a full documentation suite built with Sphinx, which can be generated and served locally.

## Features

- **Containerized Jenkins:** A pre-configured Jenkins environment using `Dockerfile` and `docker-compose.yaml`.
- **Custom Jenkins Image:** The Docker image comes with essential tools pre-installed, including Docker, Docker Compose, Python 3.10, Node.js, and Zsh.
- **Sample ML Pipeline:** Includes `train_service.jenkinsfile`, a pipeline for a hypothetical machine learning training service.
- **Makefile Automation:** A `Makefile` provides simple commands for managing the project, such as building and serving the documentation.
- **Sphinx Documentation:** A dedicated `docs/` directory with a Sphinx setup for project documentation.

## Project Structure

```
.
├── docker-compose.yaml         # Docker Compose file for the Jenkins environment.
├── Dockerfile                  # Dockerfile for the custom Jenkins image.
├── Makefile                    # Makefile with helper commands.
├── README.md                   # This README file.
├── jenkis_pipleines/
│   └── train_service.jenkinsfile # Sample Jenkins pipeline script.
└── docs/                         # Sphinx documentation directory.
    ├── Makefile
    ├── source/
    └── ...
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Build and run the Jenkins container:**
    ```bash
    docker-compose up -d --build
    ```
    This command will build the custom Jenkins image and start the container in the background.

3.  **Get the initial admin password:**
    Jenkins requires an initial password for the first login. You can retrieve it with the following command:
    ```bash
    docker exec -it jenkins cat /var/jenkins_home/secrets/initialAdminPassword
    ```

4.  **Access Jenkins:**
    Open your web browser and navigate to `http://localhost:50443`. Use the password from the previous step to log in and complete the setup.

## Usage

### Jenkins Pipeline

The core of this project is the `train_service.jenkinsfile` pipeline.

-   **Purpose**: It simulates triggering a machine learning training process.
-   **Execution**: The pipeline changes to a project directory (`/home/projects/user_train_service` inside the container) and runs a Python script (`send_to_train.py`).
-   **Configuration**: To make this pipeline fully functional, you would need to:
    1.  Create a new Pipeline job in Jenkins.
    2.  Point it to the `jenkis_pipleines/train_service.jenkinsfile` script in your SCM (Git) configuration.
    3.  Ensure the project files (like `send_to_train.py`) are present in the `/home/projects/user_train_service` directory, which is mounted from your local `./projects` folder.

### Makefile Commands

The `Makefile` provides convenient commands for managing the documentation.

-   **Build the documentation:**
    This command runs Sphinx to generate the HTML documentation from the source files in `docs/source`. The output is placed in `docs/_build/html`.
    ```bash
    make build-docs
    ```

-   **Serve the documentation:**
    This command starts a simple web server to serve the built documentation. You can view it at `http://localhost:8000`.
    ```bash
    make serve-docs
    ```

## Documentation

This project uses Sphinx for documentation.

-   **Source Files**: The documentation source is located in the `docs/source` directory. You can edit the `.rst` files there to update the content.
-   **Building**: Use the `make build-docs` command to generate the HTML output.
-   **Viewing**: Use the `make serve-docs` command to view the documentation in your browser.

## Contributing

To contribute to this project:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and test them.
4.  Submit a pull request with a clear description of your changes.

## License

[Specify the license if applicable, e.g., MIT License]