# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
from inspect import signature

__all__ = ["deprecated"]


class deprecated:
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    Examples
    --------
    >>> from sklearn.utils import deprecated
    >>> deprecated()
    <sklearn.utils.deprecation.deprecated object at ...>
    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : str, default=''
          To be added to the deprecation messages.
    """

    # Adapted from https://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        """Call method

        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        elif isinstance(obj, property):
            # Note that this is only triggered properly if the `deprecated`
            # decorator is placed before the `property` decorator, like so:
            #
            # @deprecated(msg)
            # @property
            # def deprecated_attribute_(self):
            #     ...
            return self._decorate_property(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        pass

    def _decorate_fun(self, fun):
        """Decorate function fun"""
        pass

    def _decorate_property(self, prop):
        pass


def _is_deprecated(func):
    """Helper to check if func is wrapped by our deprecated decorator"""
    pass
