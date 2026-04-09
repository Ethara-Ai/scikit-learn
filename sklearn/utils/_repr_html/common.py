# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
import inspect
import re
from functools import lru_cache
from urllib.parse import quote

from sklearn.externals._numpydoc import docscrape


def generate_link_to_param_doc(estimator_class, param_name, doc_link):
    """URL to the relevant section of the docstring using a Text Fragment

    https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Fragment/Text_fragments
    """
    pass


@lru_cache
def scrape_estimator_docstring(docstring):
    pass


def get_docstring(estimator_class, section_name, item):
    """Extract and format docstring information for a specific item.

    Parses the estimator's docstring to retrieve documentation for a
    specific parameter or attribute, formatting it as HTML-escaped text.

    Parameters
    ----------
    estimator_class : type
        The estimator class whose docstring will be parsed.

    section_name : str
        The numpydoc section to search in (e.g., "Parameters", "Attributes").

    item : str
        The name of the parameter or attribute to retrieve documentation for.

    Returns
    -------
    item_description : str or None
        HTML-formatted docstring to be used as a tooltip. Returns None if the
        estimator has no docstring or if the item is not found in the
        specified section.
    """
    pass
