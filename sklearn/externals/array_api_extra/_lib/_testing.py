"""
Testing utilities.

Note that this is private API; don't expect it to be stable.
See also ..testing for public testing utilities.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from ._utils._compat import (
    array_namespace,
    is_array_api_strict_namespace,
    is_cupy_namespace,
    is_dask_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_pydata_sparse_namespace,
    is_torch_array,
    is_torch_namespace,
    to_device,
)
from ._utils._typing import Array, Device

__all__ = ["as_numpy_array", "xp_assert_close", "xp_assert_equal", "xp_assert_less"]


def _check_ns_shape_dtype(
    actual: Array,
    desired: Array,
    check_dtype: bool,
    check_shape: bool,
    check_scalar: bool,
) -> ModuleType:  # numpydoc ignore=RT03
    """
    Assert that namespace, shape and dtype of the two arrays match.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    Returns
    -------
    Arrays namespace.
    """
    pass


def _is_materializable(x: Array) -> bool:
    """
    Return True if you can call `as_numpy_array(x)`; False otherwise.
    """
    pass


def as_numpy_array(array: Array, *, xp: ModuleType) -> np.typing.NDArray[Any]:
    """
    Convert array to NumPy, bypassing GPU-CPU transfer guards and densification guards.
    """
    pass


def xp_assert_equal(
    actual: Array,
    desired: Array,
    *,
    err_msg: str = "",
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
) -> None:
    """
    Array-API compatible version of `np.testing.assert_array_equal`.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).
    err_msg : str, optional
        Error message to display on failure.
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    See Also
    --------
    xp_assert_close : Similar function for inexact equality checks.
    numpy.testing.assert_array_equal : Similar function for NumPy arrays.
    """
    pass


def xp_assert_less(
    x: Array,
    y: Array,
    *,
    err_msg: str = "",
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
) -> None:
    """
    Array-API compatible version of `np.testing.assert_array_less`.

    Parameters
    ----------
    x, y : Array
        The arrays to compare according to ``x < y`` (elementwise).
    err_msg : str, optional
        Error message to display on failure.
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    See Also
    --------
    xp_assert_close : Similar function for inexact equality checks.
    numpy.testing.assert_array_equal : Similar function for NumPy arrays.
    """
    pass


def xp_assert_close(
    actual: Array,
    desired: Array,
    *,
    rtol: float | None = None,
    atol: float = 0,
    err_msg: str = "",
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
) -> None:
    """
    Array-API compatible version of `np.testing.assert_allclose`.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).
    rtol : float, optional
        Relative tolerance. Default: dtype-dependent.
    atol : float, optional
        Absolute tolerance. Default: 0.
    err_msg : str, optional
        Error message to display on failure.
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    See Also
    --------
    xp_assert_equal : Similar function for exact equality checks.
    isclose : Public function for checking closeness.
    numpy.testing.assert_allclose : Similar function for NumPy arrays.

    Notes
    -----
    The default `atol` and `rtol` differ from `xp.all(xpx.isclose(a, b))`.
    """
    pass


def xfail(
    request: pytest.FixtureRequest, *, reason: str, strict: bool | None = None
) -> None:
    """
    XFAIL the currently running test.

    Unlike ``pytest.xfail``, allow rest of test to execute instead of immediately
    halting it, so that it may result in a XPASS.
    xref https://github.com/pandas-dev/pandas/issues/38902

    Parameters
    ----------
    request : pytest.FixtureRequest
        ``request`` argument of the test function.
    reason : str
        Reason for the expected failure.
    strict: bool, optional
        If True, the test will be marked as failed if it passes.
        If False, the test will be marked as passed if it fails.
        Default: ``xfail_strict`` value in ``pyproject.toml``, or False if absent.
    """
    if strict is not None:
        marker = pytest.mark.xfail(reason=reason, strict=strict)
    else:
        marker = pytest.mark.xfail(reason=reason)
    request.node.add_marker(marker)
