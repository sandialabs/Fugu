import os
import sys

# from sphinx.ext.apidoc import main as sphinx_apidoc

# -- Project information -----------------------------------------------------

project = 'Fugu'
copyright = '2022, Sandia National Labs'
author = 'Sandia National Labs'

# The full version, including alpha/beta/rc tags
release = '.01'

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# sys.path.insert(0, os.path.abspath('../../examples'))
# sys.path.insert(0, os.path.abspath('../../fugu'))

for x in os.walk('../../fugu'):
    sys.path.insert(0, x[0])

for x in os.walk('../../examples'):
    sys.path.insert(0, x[0])

sys.path.insert(0, os.path.abspath('../..'))

# Enable todo items
todo_include_todos = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

##import mock
##
##MOCK_MODULES = ['numpy', 'scipy', 'sklearn', 'decorator', 'future', 'greenlet', 'msgpack', 'networkx', 'pandas', 'python', 'pytz', 'six', 'furo', 'sphinx', 'sphinx_rtd_theme']
##for mod_name in MOCK_MODULES:
##    sys.modules[mod_name] = mock.Mock()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.mock',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
]

autosummary_generate = True


# Set up autodocs options
autoclass_content = "both"
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'autosummary': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True
}

autodoc_mock_imports = ["nxsdk", 'numpy', 'scipy', 'sklearn', 'decorator', 'future', 'greenlet', 'msgpack', 'networkx', 'pandas', 'python', 'pytz', 'six', 'furo', 'sphinx', 'sphinx_rtd_theme']
### From flexpart
source_suffix = ['.rst', '.md', '.txt']

# The master toctree document.
master_doc = 'index'

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.4'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_fugu_root', '.fugu-env']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
