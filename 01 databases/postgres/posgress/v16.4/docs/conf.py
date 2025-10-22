# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Custom PostgreSQL Docker Setup'
copyright = '2025, wisrovi'
author = 'wisrovi'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True