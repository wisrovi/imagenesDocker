Installation
============

This section provides detailed instructions for installing the AI Tools Installation Scripts and the tools they manage.

Prerequisites
-------------

Before using these scripts, ensure your system meets the following requirements:

- Unix-like operating system (Linux, macOS, or WSL on Windows)
- Bash shell
- npm (for Google Gemini CLI installation)
- curl (for OpenCode installation)

Installing Individual Tools
---------------------------

Google Gemini CLI
~~~~~~~~~~~~~~~~~~

The Google Gemini CLI is an AI model developed by Google for various natural language processing tasks.

To install:

.. code-block:: bash

   ./install/gemini.sh

This script executes:

.. code-block:: bash

   npm install -g @google/gemini-cli

OpenCode
~~~~~~~~

OpenCode is an AI-powered code assistance tool.

To install:

.. code-block:: bash

   ./install/opencode.sh

This script executes:

.. code-block:: bash

   curl -fsSL https://opencode.ai/install | bash

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

- For Google Gemini CLI: Run ``gemini --help``
- For OpenCode: Check the OpenCode documentation for verification steps

Troubleshooting
---------------

Common issues:

- **Permission denied**: Run with ``sudo`` if installing globally
- **npm not found**: Install Node.js and npm first
- **curl not found**: Install curl using your package manager

For more help, refer to the official documentation of each tool.