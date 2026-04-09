# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
import reprlib
from collections import UserDict

from sklearn.utils._repr_html.base import ReprHTMLMixin
from sklearn.utils._repr_html.common import (
    generate_link_to_param_doc,
    get_docstring,
)


def _read_params(name, value, non_default_params):
    """Categorizes parameters as 'default' or 'user-set' and formats their values.
    Escapes or truncates parameter values for display safety and readability.
    """
    pass


def _params_html_repr(params):
    """Generate HTML representation of estimator parameters.

    Creates an HTML table with parameter names and values, wrapped in a
    collapsible details element. Parameters are styled differently based
    on whether they are default or user-set values.
    """
    pass


class ParamsDict(ReprHTMLMixin, UserDict):
    """Dictionary-like class to store and provide an HTML representation.

    It builds an HTML structure to be used with Jupyter notebooks or similar
    environments. It allows storing metadata to track non-default parameters.

    Parameters
    ----------
    params : dict, default=None
        The original dictionary of parameters and their values.

    non_default : tuple, default=(,)
        The list of non-default parameters.

    estimator_class : type, default=None
        The class of the estimator. It allows to find the online documentation
        link for each parameter.

    doc_link : str, default=""
        The base URL to the online documentation for the estimator class.
        Used to generate parameter-specific documentation links in the HTML
        representation. If empty, documentation links will not be generated.
    """

    _html_repr = _params_html_repr

    def __init__(
        self, *, params=None, non_default=tuple(), estimator_class=None, doc_link=""
    ):
        super().__init__(params or {})
        self.non_default = non_default
        self.estimator_class = estimator_class
        self.doc_link = doc_link
