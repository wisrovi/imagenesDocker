# PR Analyzer

A tool to analyze open Pull Requests in a GitHub repository. It checks for approvals, requested changes, and runs Pylint to calculate a code quality score.

## Features

- Clones a GitHub repository specified at runtime.
- Iterates through all open Pull Requests.
- For each PR, it checks out the source branch.
- Analyzes the PR for reviewer approvals and requested changes.
- Runs Pylint on all Python files in the PR.
- Calculates an average Pylint score.
- Saves a consolidated JSON report of all analyzed PRs.
- Prints a final summary to the console.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Docker:** The tool runs in a Docker container. Make sure Docker is installed and running.
2.  **Git:** The tool uses `git` to clone repositories.
3.  **SSH Key:** An SSH key properly configured in your `~/.ssh/` directory and added to your GitHub account. This is required for cloning private repositories.
4.  **GitHub Personal Access Token (PAT):** The tool uses a PAT to interact with the GitHub API.

## Setup

1.  **Secrets Configuration:**

    Create a `config/secrets.env` file. This file should contain your GitHub Personal Access Token.

    ```
    # config/secrets.env
    GITHUB_PAT="your_personal_access_token_here"
    ```
    *(Note: `config/secrets.env` is ignored by Git by default)*

2.  **Build the Docker Image:**

    Run the build script. This will create a generic Docker image named `wisrovi/pr_analizer:latest` that can be used to analyze any repository.

    ```bash
    sh scripts/build_image.sh
    ```

## Usage

To run the analysis, execute the `run_analysis.sh` script, passing the repository owner and repository name as arguments.

**Syntax:**
```bash
sh scripts/run_analysis.sh <repository_owner> <repository_name>
```

**Example:**
```bash
sh scripts/run_analysis.sh my-organization my-awesome-repo
```

The script will:
- Pass the repository owner and name to the Docker container as environment variables.
- Mount your `~/.ssh` directory for Git authentication.
- Mount the `config` directory to provide the secrets to the container.
- Mount the `report` directory for saving the output file.
- Print a final analysis summary to the console.
- Save a consolidated JSON report of all analyzed PRs.

This method is ideal for CI/CD environments like Jenkins, as it allows you to use the same Docker image to analyze different repositories dynamically.

## Reports

A single, consolidated JSON report named `report.json` is generated, containing the analysis for all open Pull Requests.

The report is saved in the `report/` directory, following this structure:

```
report/
└── <repository_owner>/
    └── <repository_name>/
        └── report.json
```

The JSON file contains a dictionary where each key is a PR number and the value is the detailed analysis for that PR. The file is overwritten on each run.

## Testing

This project uses `pytest` for unit testing and `pytest-cov` for code coverage. All tests are executed inside a Docker container to ensure a consistent and isolated environment.

1.  **Build the Image:**
    Before running the tests, ensure you have an up-to-date Docker image with your latest code changes. If you've made any changes, rebuild the image:
    ```bash
    sh scripts/build_image.sh
    ```

2.  **Run Tests:**
    Execute the `run_tests.sh` script:
    ```bash
    sh scripts/run_tests.sh
    ```
    This will start a new container from your project's image, run all unit tests using `pytest`, and generate a code coverage report in the terminal.

## Project Structure

The project is structured as an installable Python library:

```
.
|-- Dockerfile             # Defines the container environment
|-- config/                # Project configuration
|   |-- secrets.env
|-- pyproject.toml         # Build system configuration
|-- requirements.txt       # Python dependencies
|-- scripts/               # Build and run scripts
|   |-- build_image.sh
|   |-- run_analysis.sh
|   |-- run_tests.sh
|-- setup.cfg              # Package configuration (name, entry points)
|-- src/
|   |-- pr_analyzer/       # Main library source code
|   |   |-- __init__.py
|   |   |-- analysis.py      # Pylint analysis logic
|   |   |-- config.py        # Internal configuration constants
|   |   |-- github_client.py # Handles GitHub API communication
|   |   |-- main.py          # Main application entry point
|-- tests/                 # Unit tests
|   |-- test_analysis.py
|   |-- test_github_client.py
```