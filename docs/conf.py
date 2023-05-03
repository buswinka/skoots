import os.path
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "skoots"
copyright = "2023, Chris Buswinka"
author = "Chris Buswinka"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath('..'))

master_doc = 'index'

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",  # use :ref:`Heading` for any heading
    "myst_nb",
    # Our custom extension, only meant for Furo's own documentation.
    "furo.sphinxext",
    # External stuff
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = [
    "build",
    "_build",
    "**.ipynb_checkpoints",
    "links.rst",
    "sinebow.rst",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

autoclass_content = "init"

# exclude_patterns = ['sinebow.rst']


# from omnipose. Thanks!
import math


def rgb_2_hex(rgb):
    return f"#{hex(int(rgb[0]))[-2::]}{hex(int(rgb[1]))[-2::]}{hex(int(rgb[2]))[-2::]}"


def sinebow(N, bg_color=[0, 0, 0, 0]):
    """Generate a color dictionary for use in visualizing N-colored labels. Background color
    defaults to transparent black.

    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)

    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the  dictionary.

    Returns
    --------------
    Dictionary with entries {int:RGBA array} to map integer labels to RGBA colors.

    """
    colordict = {0: bg_color}
    for j in range(N):
        angle = j * 2 * math.pi / (N)
        r = (math.cos(angle) + 1) / 2
        g = (math.cos(angle + 2 * math.pi / 3) + 1) / 2
        b = (math.cos(angle + 4 * math.pi / 3) + 1) / 2
        colordict.update(
            {j + 1: [int(r * 255), int(g * 255), int(b * 255), int(1 * 255)]}
        )
    return colordict


N = 42
colors = sinebow(N)
colors = [rgb_2_hex(colors[i]) for i in range(1, N + 1)]
colordict = {}
for i in range(N):
    colordict["sinebow" + "%0d" % i] = colors[i]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "furo"
html_static_path = ["_static"]

templates_path = ["_templates"]
html_show_sphinx = False

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "blue",
        "color-brand-content": "#CC3333",
        "color-admonition-background": "orange",
        "sinebow0": "#CC3333",
    },
    "sidebar_hide_name": True,
    "top_of_page_button": "edit",
    "dark_css_variables": colordict,
}

html_logo = "../resources/skooting_in_progress.png"
