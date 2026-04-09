from functools import cache

from sphinx.util.logging import getLogger

logger = getLogger(__name__)


def override_pst_pagetoc(app, pagename, templatename, context, doctree):
    """Overrides the `generate_toc_html` function of pydata-sphinx-theme for API."""
    pass


def setup(app):
    # Need to be triggered after `pydata_sphinx_theme.toctree.add_toctree_functions`,
    # and since default priority is 500 we set 900 for safety
    app.connect("html-page-context", override_pst_pagetoc, priority=900)
