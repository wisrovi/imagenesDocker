# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'MCP Inspector Docker Setup'
copyright = '2025, William Rodriguez'
author = 'William Rodriguez'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (root_doc, 'MCPInspectorDockerSetup.tex', 'MCP Inspector Docker Setup Documentation',
     'William Rodriguez', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (root_doc, 'mcpinspectordockersetup', 'MCP Inspector Docker Setup Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (root_doc, 'MCPInspectorDockerSetup', 'MCP Inspector Docker Setup Documentation',
     author, 'MCPInspectorDockerSetup', 'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration --------------------------------------------------
root_doc = 'index'