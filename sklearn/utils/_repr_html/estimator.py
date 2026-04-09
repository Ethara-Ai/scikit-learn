# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path

from sklearn import config_context


class _IDCounter:
    """Generate sequential ids with a prefix."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.count = 0

    def get_id(self):
        pass


def _get_css_style():
    estimator_css_file = Path(__file__).parent / "estimator.css"
    params_css_file = Path(__file__).parent / "params.css"

    estimator_css = estimator_css_file.read_text(encoding="utf-8")
    params_css = params_css_file.read_text(encoding="utf-8")

    return f"{estimator_css}\n{params_css}"


_CONTAINER_ID_COUNTER = _IDCounter("sk-container-id")
_ESTIMATOR_ID_COUNTER = _IDCounter("sk-estimator-id")
_CSS_STYLE = _get_css_style()


class _VisualBlock:
    """HTML Representation of Estimator

    Parameters
    ----------
    kind : {'serial', 'parallel', 'single'}
        kind of HTML block

    estimators : list of estimators or `_VisualBlock`s or a single estimator
        If kind != 'single', then `estimators` is a list of
        estimators.
        If kind == 'single', then `estimators` is a single estimator.

    names : list of str, default=None
        If kind != 'single', then `names` corresponds to estimators.
        If kind == 'single', then `names` is a single string corresponding to
        the single estimator.

    name_details : list of str, str, or None, default=None
        If kind != 'single', then `name_details` corresponds to `names`.
        If kind == 'single', then `name_details` is a single string
        corresponding to the single estimator.

    name_caption : str, default=None
        The caption below the name. `None` stands for no caption.
        Only active when kind == 'single'.

    doc_link_label : str, default=None
        The label for the documentation link. If provided, the label would be
        "Documentation for {doc_link_label}". Otherwise it will look for `names`.
        Only active when kind == 'single'.

    dash_wrapped : bool, default=True
        If true, wrapped HTML element will be wrapped with a dashed border.
        Only active when kind != 'single'.
    """

    def __init__(
        self,
        kind,
        estimators,
        *,
        names=None,
        name_details=None,
        name_caption=None,
        doc_link_label=None,
        dash_wrapped=True,
    ):
        self.kind = kind
        self.estimators = estimators
        self.dash_wrapped = dash_wrapped
        self.name_caption = name_caption
        self.doc_link_label = doc_link_label

        if self.kind in ("parallel", "serial"):
            if names is None:
                names = (None,) * len(estimators)
            if name_details is None:
                name_details = (None,) * len(estimators)

        self.names = names
        self.name_details = name_details

    def _sk_visual_block_(self):
        pass


def _write_label_html(
    out,
    params,
    attrs,
    name,
    name_details,
    name_caption=None,
    doc_link_label=None,
    outer_class="sk-label-container",
    inner_class="sk-label",
    checked=False,
    doc_link="",
    is_fitted_css_class="",
    is_fitted_icon="",
    param_prefix="",
):
    """Write labeled html with or without a dropdown with named details.

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    params: str
        If estimator has `get_params` method, this is the HTML representation
        of the estimator's parameters and their values. When the estimator
        does not have `get_params`, it is an empty string.
    attrs: str
        If estimator is fitted, this is the HTML representation of its
        fitted attributes.
    name : str
        The label for the estimator. It corresponds either to the estimator class name
        for a simple estimator or in the case of a `Pipeline` and `ColumnTransformer`,
        it corresponds to the name of the step.
    name_details : str
        The details to show as content in the dropdown part of the toggleable label. It
        can contain information such as non-default parameters or column information for
        `ColumnTransformer`.
    name_caption : str, default=None
        The caption below the name. If `None`, no caption will be created.
    doc_link_label : str, default=None
        The label for the documentation link. If provided, the label would be
        "Documentation for {doc_link_label}". Otherwise it will look for `name`.
    outer_class : {"sk-label-container", "sk-item"}, default="sk-label-container"
        The CSS class for the outer container.
    inner_class : {"sk-label", "sk-estimator"}, default="sk-label"
        The CSS class for the inner container.
    checked : bool, default=False
        Whether the dropdown is folded or not. With a single estimator, we intend to
        unfold the content.
    doc_link : str, default=""
        The link to the documentation for the estimator. If an empty string, no link is
        added to the diagram. This can be generated for an estimator if it uses the
        `_HTMLDocumentationLinkMixin`.
    is_fitted_css_class : {"", "fitted"}
        The CSS class to indicate whether or not the estimator is fitted. The
        empty string means that the estimator is not fitted and "fitted" means that the
        estimator is fitted.
    is_fitted_icon : str, default=""
        The HTML representation to show the fitted information in the diagram. An empty
        string means that no information is shown.
    param_prefix : str, default=""
        The prefix to prepend to parameter names for nested estimators.
    """
    pass


def _get_visual_block(estimator):
    """Generate information about how to display an estimator."""
    pass


def _write_estimator_html(
    out,
    estimator,
    estimator_label,
    estimator_label_details,
    is_fitted_css_class,
    is_fitted_icon="",
    first_call=False,
    param_prefix="",
):
    """Write estimator to html in serial, parallel, or by itself (single).

    For multiple estimators, this function is called recursively.

    Parameters
    ----------
    out : file-like object
        The file to write the HTML representation to.
    estimator : estimator object
        The estimator to visualize.
    estimator_label : str
        The label for the estimator. It corresponds either to the estimator class name
        for simple estimator or in the case of `Pipeline` and `ColumnTransformer`, it
        corresponds to the name of the step.
    estimator_label_details : str
        The details to show as content in the dropdown part of the toggleable label.
        It can contain information as non-default parameters or column information for
        `ColumnTransformer`.
    is_fitted_css_class : {"", "fitted"}
        The CSS class to indicate whether or not the estimator is fitted or not. The
        empty string means that the estimator is not fitted and "fitted" means that the
        estimator is fitted.
    is_fitted_icon : str, default=""
        The HTML representation to show the fitted information in the diagram. An empty
        string means that no information is shown. If the estimator to be shown is not
        the first estimator (i.e. `first_call=False`), `is_fitted_icon` is always an
        empty string.
    first_call : bool, default=False
        Whether this is the first time this function is called.
    param_prefix : str, default=""
        The prefix to prepend to parameter names for nested estimators.
        For example, in a pipeline this might be "pipeline__stepname__".
    """
    pass


def estimator_html_repr(estimator):
    """Build an HTML representation of an estimator.

    Read more in the :ref:`User Guide <visualizing_composite_estimators>`.

    Parameters
    ----------
    estimator : estimator object
        The estimator to visualize.

    Returns
    -------
    html: str
        HTML representation of estimator.

    Examples
    --------
    >>> from sklearn.utils._repr_html.estimator import estimator_html_repr
    >>> from sklearn.linear_model import LogisticRegression
    >>> estimator_html_repr(LogisticRegression())
    '<style>.sk-global...'
    """
    pass
