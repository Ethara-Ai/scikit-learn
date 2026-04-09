"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import platform
import sys

from threadpoolctl import threadpool_info

from sklearn import __version__
from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    pass


def _get_deps_info():
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    pass


def show_versions():
    """Print useful debugging information.

    .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn import show_versions
    >>> show_versions()  # doctest: +SKIP
    """
    pass
