# AI Tools Installation Scripts

## Overview

This repository provides a collection of installation scripts and templates for setting up various AI-related tools and utilities. It is designed to simplify the deployment and configuration of popular AI development environments, including command-line interfaces for AI services and automation tools.

The project focuses on streamlining the setup process for developers and researchers working with artificial intelligence technologies, ensuring quick and reliable installation of essential tools.

## Contents

### Directory Structure

```
agents/
├── docs/                  # Sphinx documentation
│   ├── _build/            # Built documentation (generated)
│   ├── _static/           # Static files for docs
│   ├── author.rst         # Author information
│   ├── conf.py            # Sphinx configuration
│   ├── index.rst          # Main documentation page
│   ├── installation.rst   # Installation guide
│   ├── templates.rst      # Templates documentation
│   ├── usage.rst          # Usage guide
│   └── Makefile           # Documentation build commands
├── install/
│   ├── base.sh            # Installation script for Node.js via nvm
│   ├── copilot.sh         # Installation script for GitHub Copilot CLI
│   ├── gemini.sh          # Installation script for Google Gemini CLI
│   └── opencode.sh        # Installation script for OpenCode
├── Makefile               # Project build and installation commands
├── README.md              # This file
└── templates.yaml         # YAML file containing task templates
```

### Files Description

- **docs/**: Contains the Sphinx-generated documentation for the project, including installation guides, usage instructions, and author information.

- **install/base.sh**: A shell script that installs Node Version Manager (nvm) and sets up Node.js version 22 for development environments. Includes inline comments explaining each command.

- **install/copilot.sh**: A shell script that installs GitHub Copilot CLI globally using npm for AI-powered code assistance in the terminal. Requires Node.js and npm.

- **install/gemini.sh**: A shell script that installs Google Gemini CLI globally using npm for interacting with Google's Gemini AI models. Requires Node.js and npm.

- **install/opencode.sh**: A shell script that installs the opencode CLI tool from the official installer for AI-assisted coding. Downloads and executes the installer script.

- **Makefile**: The main Makefile for the project, providing commands to install tools, build documentation, and manage the project.

- **templates.yaml**: A YAML configuration file containing predefined templates for common development tasks. Currently includes a template for creating professional README documentation.

## Quick Start

To quickly set up the entire project:

```bash
make all
```

This command will install all AI tools and build the documentation.

## Installation

### Prerequisites

- Unix-like operating system (Linux, macOS)
- Bash shell
- curl (for base.sh, opencode.sh)
- **Note**: Installing nvm and npm packages globally may require sudo privileges on some systems. Ensure you have administrative access or run the scripts with appropriate permissions.

### Using Makefile Commands

The project includes a Makefile for easy management:

- `make install`: Install all AI tools
- `make install-gemini`: Install only Google Gemini CLI
- `make install-base`: Install Node.js via nvm
- `make install-copilot`: Install GitHub Copilot CLI
- `make install-opencode`: Install only OpenCode
- `make docs`: Build the documentation
- `make serve-docs`: Build and serve documentation locally

### Installing Individual Tools

#### Base Installation (Node.js)

To install Node.js via nvm:

```bash
./install/base.sh
```

This script downloads and installs Node Version Manager (nvm), sources the shell configuration, verifies the installation, installs Node.js version 22, and sets it as the active version. **Note**: May require sudo for global npm installations in subsequent scripts.

#### GitHub Copilot CLI

To install the GitHub Copilot CLI:

```bash
./install/copilot.sh
```

This script installs GitHub Copilot CLI globally using npm. **Note**: Requires Node.js and npm (install base.sh first). May require sudo for global npm installation.

#### Google Gemini CLI

To install the Google Gemini CLI:

```bash
./install/gemini.sh
```

This script installs Google Gemini CLI globally using npm for interacting with Gemini AI models. **Note**: Requires Node.js and npm (install base.sh first). May require sudo for global npm installation.

#### OpenCode

To install OpenCode:

```bash
./install/opencode.sh
```

This script downloads and executes the official opencode installation script from opencode.ai. **Note**: Requires curl and an internet connection. May require sudo depending on the installer.

This script runs:
```bash
curl -fsSL https://opencode.ai/install | bash
```

### Bulk Installation

You can run all installation scripts sequentially:

```bash
for script in install/*.sh; do
    echo "Running $script..."
    bash "$script"
done
```

## Usage

After installation, you can use the installed tools according to their respective documentation:

- **GitHub Copilot CLI**: Refer to the official GitHub Copilot documentation for usage instructions.
- **Google Gemini CLI**: Refer to the official Google Gemini documentation for usage instructions.
- **OpenCode**: Check the OpenCode documentation at https://opencode.ai for detailed usage guides.

## Documentation

The project includes comprehensive documentation built with Sphinx. To build and view the documentation:

```bash
make docs
make serve-docs
```

The documentation covers installation, usage, templates, and author information. Once served, access it at http://localhost:8000.

## Templates

The `templates.yaml` file contains reusable templates for common tasks. Each template includes:

- **name**: A unique identifier for the template
- **description**: A detailed description of what the template accomplishes

### Current Templates

1. **CREAR_README**: A template for creating professional README documentation in English. It involves analyzing the project structure, understanding the codebase, and generating comprehensive documentation.

### Using Templates

Templates can be used as starting points for various development tasks. To apply a template:

1. Review the template description in `templates.yaml`
2. Adapt the template to your specific needs
3. Execute the described steps

## Contributing

Contributions to this repository are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add your installation scripts or templates
4. Test your changes thoroughly
5. Submit a pull request with a clear description of your changes

Please ensure that:
- Installation scripts are tested on multiple platforms
- Templates are well-documented
- Code follows best practices for shell scripting

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

If you encounter issues with installation or usage:

1. Check the official documentation for each tool
2. Verify system requirements
3. Open an issue in this repository with detailed error messages and system information

## Version History

- v1.2.0: Added base.sh for Node.js/nvm installation, copilot.sh for GitHub Copilot CLI, enhanced docstrings and inline comments in all scripts, updated README and Makefile accordingly.
- v1.1.0: Added Sphinx documentation, Makefile for project management, and expanded README.
- v1.0.0: Initial release with Gemini CLI and OpenCode installation scripts, plus basic templates.