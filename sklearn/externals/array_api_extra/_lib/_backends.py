"""Backends against which array-api-extra runs its tests."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pytest

__all__ = ["NUMPY_VERSION", "Backend"]

NUMPY_VERSION = tuple(int(v) for v in np.__version__.split(".")[:3])  # pyright: ignore[reportUnknownArgumentType]


class Backend(Enum):  # numpydoc ignore=PR02
    """
    All array library backends explicitly tested by array-api-extra.

    Parameters
    ----------
    value : str
        Tag of the backend's module, in the format ``<namespace>[:<extra tag>]``.
    """

    # Use :<tag> to prevent Enum from deduplicating items with the same value
    ARRAY_API_STRICT = "array_api_strict"
    ARRAY_API_STRICTEST = "array_api_strict:strictest"
    NUMPY = "numpy"
    NUMPY_READONLY = "numpy:readonly"
    CUPY = "cupy"
    TORCH = "torch"
    TORCH_GPU = "torch:gpu"
    DASK = "dask.array"
    SPARSE = "sparse"
    JAX = "jax.numpy"
    JAX_GPU = "jax.numpy:gpu"

    @property
    def modname(self) -> str:  # numpydoc ignore=RT01
        """Module name to be imported."""
        pass

    def like(self, *others: Backend) -> bool:  # numpydoc ignore=PR01,RT01
        """Check if this backend uses the same module as others."""
        pass

    def pytest_param(self) -> Any:
        """
        Backend as a pytest parameter

        Returns
        -------
        pytest.mark.ParameterSet
        """
        pass
