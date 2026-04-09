"""
doilinks
~~~~~~~~
Extension to add links to DOIs. With this extension you can use e.g.
:doi:`10.1016/S0022-2836(05)80360-2` in your documents. This will
create a link to a DOI resolver
(``https://doi.org/10.1016/S0022-2836(05)80360-2``).
The link caption will be the raw DOI.
You can also give an explicit caption, e.g.
:doi:`Basic local alignment search tool <10.1016/S0022-2836(05)80360-2>`.

:copyright: Copyright 2015  Jon Lund Steffensen. Based on extlinks by
    the Sphinx team.
:license: BSD.
"""

from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title


def reference_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    pass


def setup_link_role(app):
    pass


def setup(app):
    app.connect("builder-inited", setup_link_role)
    return {"version": "0.1", "parallel_read_safe": True}
