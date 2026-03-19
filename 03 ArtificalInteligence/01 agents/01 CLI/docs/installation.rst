Installation
============

This section provides detailed instructions for installing the AI Tools Installation Scripts and the tools they manage.

Prerequisites
-------------

Before using these scripts, ensure your system meets the following requirements:

- Unix-like operating system (Linux, macOS, or WSL on Windows)
- Bash shell
- curl (for base.sh, opencode.sh)
- **Note**: Installing nvm and npm packages globally may require sudo privileges on some systems. Ensure you have administrative access or run the scripts with appropriate permissions.

Installing Individual Tools
---------------------------

Base Installation (Node.js)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base installation script sets up Node Version Manager (nvm) and Node.js version 22 for development environments.

To install:

.. code-block:: bash

   ./install/base.sh

This script downloads and installs nvm, sources the shell configuration, verifies installation, installs Node.js 22, and sets it as active.

GitHub Copilot CLI
~~~~~~~~~~~~~~~~~~~

GitHub Copilot CLI provides AI-powered code assistance in the terminal.

To install:

.. code-block:: bash

   ./install/copilot.sh

This script executes:

.. code-block:: bash

   npm install -g @github/copilot

**Note**: Requires Node.js and npm (install base.sh first).

Google Gemini CLI
~~~~~~~~~~~~~~~~~~

The Google Gemini CLI is an AI model developed by Google for various natural language processing tasks.

To install:

.. code-block:: bash

    ./install/gemini.sh

This script executes:

.. code-block:: bash

    npm install -g @google/gemini-cli

**Note**: Requires Node.js and npm (install base.sh first).

OpenCode
~~~~~~~~

OpenCode is an AI-powered code assistance tool.

To install:

.. code-block:: bash

    ./install/opencode.sh

This script executes:

.. code-block:: bash

    curl -fsSL https://opencode.ai/install | bash

**Note**: Requires curl and an internet connection.

Bulk Installation
-----------------

To install all tools at once:

.. code-block:: bash

   for script in install/*.sh; do
       echo "Running $script..."
       bash "$script"
   done

Verification
------------

After installation, verify the tools are working:

- For Node.js/nvm: Run ``node --version`` and ``npm --version``
- For GitHub Copilot CLI: Run ``gh copilot --help``
- For Google Gemini CLI: Run ``gemini --help``
- For OpenCode: Check the OpenCode documentation for verification steps

Troubleshooting
---------------

Common issues:

- **Permission denied**: Run with ``sudo`` if installing globally (especially for npm packages)
- **npm not found**: Install Node.js and npm first (run base.sh)
- **curl not found**: Install curl using your package manager
- **nvm not sourced**: Restart your terminal or source ~/.bashrc after installing nvm
- **GitHub authentication required**: For Copilot CLI, authenticate with GitHub

For more help, refer to the official documentation of each tool.