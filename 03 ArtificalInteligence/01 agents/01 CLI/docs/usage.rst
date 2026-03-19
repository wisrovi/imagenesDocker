Usage
=====

This section explains how to use the installed AI tools and the templates provided in this repository.

Using Installed Tools
---------------------

Google Gemini CLI
~~~~~~~~~~~~~~~~~~

After installation, you can use the Google Gemini CLI for various AI tasks. Refer to the official Google Gemini documentation for detailed usage instructions.

Basic usage example:

.. code-block:: bash

   gemini "Generate a summary of this text: [your text here]"

OpenCode
~~~~~~~~

OpenCode provides AI-powered code assistance. Check the OpenCode documentation at https://opencode.ai for detailed usage guides and examples.

Using Templates
---------------

The ``templates.yaml`` file contains predefined templates for common development tasks.

Template Structure
~~~~~~~~~~~~~~~~~~

Each template in ``templates.yaml`` has the following structure:

- **name**: A unique identifier for the template
- **description**: A detailed description of what the template accomplishes

Available Templates
~~~~~~~~~~~~~~~~~~~

CREAR_README
^^^^^^^^^^^^

This template is designed for creating professional README documentation in English. It involves:

1. Analyzing the project structure
2. Understanding the codebase
3. Generating comprehensive documentation
4. Updating the README file

To apply this template:

1. Review the template description in ``templates.yaml``
2. Adapt the template to your specific project needs
3. Execute the described steps

Example Usage in Automation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Templates can be used as starting points for various development tasks. For example, you could create a script that reads the templates and applies them automatically.

.. code-block:: python

   import yaml

   with open('templates.yaml', 'r') as file:
       templates = yaml.safe_load(file)

   for template in templates['templates']:
       print(f"Template: {template['name']}")
       print(f"Description: {template['description']}")
       # Apply template logic here

Best Practices
--------------

- Always test templates in a development environment first
- Customize templates to fit your specific project requirements
- Keep templates updated as your project evolves
- Document any modifications made to templates