# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
import reprlib
from collections import UserDict

import numpy as np

from sklearn.utils._repr_html.base import ReprHTMLMixin
from sklearn.utils._repr_html.common import (
    generate_link_to_param_doc,
    get_docstring,
)


def _read_fitted_attr(value):
    pass


def _fitted_attr_html_repr(fitted_attributes):
    """Generate HTML representation of estimator fitted attributes.

    Creates an HTML table with fitted attribute names and values
    wrapped in a collapsible details element. When attributes are arrays,
    shape and dtype are shown.
    """
    pass


class AttrsDict(ReprHTMLMixin, UserDict):
    """Dictionary-like class to store and provide an HTML representation.

    It builds an HTML structure to be used with Jupyter notebooks or similar
    environments.

    Parameters
    ----------
    fitted_attrs : dict, default=None
        Dictionary of fitted attributes and their values.

    estimator_class : type, default=None
        The class of the estimator. It allows to find the online documentation
        link for each parameter.

    doc_link : str, default=""
        The base URL to the online documentation for the estimator class.
        Used to generate parameter-specific documentation links in the HTML
        representation. If empty, documentation links will not be generated.
    """

    _html_repr = _fitted_attr_html_repr

    def __init__(self, *, fitted_attrs=None, estimator_class=None, doc_link=""):
        super().__init__(fitted_attrs or {})
        self.estimator_class = estimator_class
        self.doc_link = doc_link
