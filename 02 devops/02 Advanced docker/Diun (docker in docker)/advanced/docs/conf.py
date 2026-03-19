# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Docker-in-Docker with Portainer and SSH'
copyright = '2025, Wisrovi Rodriguez'
author = 'Wisrovi Rodriguez'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = []
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Enable search
html_search_language = 'es'
html_search_options = {
    'type': 'default',
    'dict': '/_static/searchdict.js'
}

# Search settings
html_search_scorer = '/_static/scorer.js'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'prev_next_buttons_location': 'both',
    'style_nav_header_background': '#2980B9',
}

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'docker': ('https://docs.docker.com', None),
}

# Todo extension
todo_include_todos = True

# Autodoc settings
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']
autodoc_member_order = 'bysource'

# Viewcode extension
viewcode_follow_imported_members = True

# Language
language = 'es'