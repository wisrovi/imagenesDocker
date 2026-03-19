Templates
=========

The ``templates.yaml`` file contains reusable templates for common development tasks in AI projects.

Overview
--------

Templates provide standardized approaches to repetitive tasks, ensuring consistency and efficiency in project development.

Current Templates
-----------------

CREAR_README
~~~~~~~~~~~~

**Purpose**: Create professional README documentation in English.

**Description**: This template guides the process of analyzing a project folder, understanding its contents, and generating comprehensive documentation. It includes steps for:

- Exploring the project structure
- Reading and analyzing all files
- Understanding the project's purpose and functionality
- Creating detailed, explanatory documentation
- Updating the README file with professional formatting

**Steps**:

1. Examine all files and directories in the project
2. Read key files to understand functionality
3. Identify the project's main purpose and components
4. Generate documentation covering overview, installation, usage, etc.
5. Update the README.md file

**Benefits**:

- Ensures consistent documentation quality
- Saves time on documentation creation
- Provides a structured approach to project analysis

Template File Structure
-----------------------

The templates are stored in YAML format for easy parsing and modification:

.. code-block:: yaml

   templates:
     - name: TEMPLATE_NAME
       description: "Detailed description of what the template does"

Extending Templates
-------------------

To add new templates:

1. Open ``templates.yaml``
2. Add a new entry under the ``templates`` list
3. Provide a unique name and detailed description
4. Test the template in your workflow

Template Validation
-------------------

Before applying templates:

- Ensure all prerequisites are met
- Verify file paths and permissions
- Test in a safe environment
- Review generated output for accuracy

Future Templates
----------------

Planned templates include:

- Automated testing setup
- CI/CD pipeline configuration
- Code quality checks
- Deployment scripts