from __future__ import annotations

import math
from typing import Literal, NamedTuple, cast

import numpy as np

if np.__version__[0] == "2":
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_tuple  # type: ignore[no-redef]

from .._internal import get_xp
from ._aliases import isdtype, matmul, matrix_transpose, tensordot, vecdot
from ._typing import Array, DType, JustFloat, JustInt, Namespace


# These are in the main NumPy namespace but not in numpy.linalg
def cross(
    x1: Array,
    x2: Array,
    /,
    xp: Namespace,
    *,
    axis: int = -1,
    **kwargs: object,
) -> Array:
    pass

def outer(x1: Array, x2: Array, /, xp: Namespace, **kwargs: object) -> Array:
    return xp.outer(x1, x2, **kwargs)

class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array

class QRResult(NamedTuple):
    Q: Array
    R: Array

class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array

class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array

# These functions are the same as their NumPy counterparts except they return
# a namedtuple.
def eigh(x: Array, /, xp: Namespace, **kwargs: object) -> EighResult:
    return EighResult(*xp.linalg.eigh(x, **kwargs))

def qr(
    x: Array,
    /,
    xp: Namespace,
    *,
    mode: Literal["reduced", "complete"] = "reduced",
    **kwargs: object,
) -> QRResult:
    return QRResult(*xp.linalg.qr(x, mode=mode, **kwargs))

def slogdet(x: Array, /, xp: Namespace, **kwargs: object) -> SlogdetResult:
    return SlogdetResult(*xp.linalg.slogdet(x, **kwargs))

def svd(
    x: Array,
    /,
    xp: Namespace,
    *,
    full_matrices: bool = True,
    **kwargs: object,
) -> SVDResult:
    return SVDResult(*xp.linalg.svd(x, full_matrices=full_matrices, **kwargs))

# These functions have additional keyword arguments

# The upper keyword argument is new from NumPy
def cholesky(
    x: Array,
    /,
    xp: Namespace,
    *,
    upper: bool = False,
    **kwargs: object,
) -> Array:
    L = xp.linalg.cholesky(x, **kwargs)
    if upper:
        U = get_xp(xp)(matrix_transpose)(L)
        if get_xp(xp)(isdtype)(U.dtype, 'complex floating'):
            U = xp.conj(U)  # pyright: ignore[reportConstantRedefinition]
        return U
    return L

# The rtol keyword argument of matrix_rank() and pinv() is new from NumPy.
# Note that it has a different semantic meaning from tol and rcond.
def matrix_rank(
    x: Array,
    /,
    xp: Namespace,
    *,
    rtol: float | Array | None = None,
    **kwargs: object,
) -> Array:
    # this is different from xp.linalg.matrix_rank, which supports 1
    # dimensional arrays.
    pass

def pinv(
    x: Array,
    /,
    xp: Namespace,
    *,
    rtol: float | Array | None = None,
    **kwargs: object,
) -> Array:
    # this is different from xp.linalg.pinv, which does not multiply the
    # default tolerance by max(M, N).
    if rtol is None:
        rtol = max(x.shape[-2:]) * xp.finfo(x.dtype).eps
    return xp.linalg.pinv(x, rcond=rtol, **kwargs)

# These functions are new in the array API spec

def matrix_norm(
    x: Array,
    /,
    xp: Namespace,
    *,
    keepdims: bool = False,
    ord: Literal[1, 2, -1, -2] | JustFloat | Literal["fro", "nuc"] | None = "fro",
) -> Array:
    pass

# svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# xp.linalg.svd(compute_uv=False).
def svdvals(x: Array, /, xp: Namespace) -> Array | tuple[Array, ...]:
    return xp.linalg.svd(x, compute_uv=False)

def vector_norm(
    x: Array,
    /,
    xp: Namespace,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: JustInt | JustFloat = 2,
) -> Array:
    # xp.linalg.norm tries to do a matrix norm whenever axis is a 2-tuple or
    # when axis=None and the input is 2-D, so to force a vector norm, we make
    # it so the input is 1-D (for axis=None), or reshape so that norm is done
    # on a single dimension.
    pass

# xp.diagonal and xp.trace operate on the first two axes whereas these
# operates on the last two

def diagonal(x: Array, /, xp: Namespace, *, offset: int = 0, **kwargs: object) -> Array:
    return xp.diagonal(x, offset=offset, axis1=-2, axis2=-1, **kwargs)

def trace(
    x: Array,
    /,
    xp: Namespace,
    *,
    offset: int = 0,
    dtype: DType | None = None,
    **kwargs: object,
) -> Array:
    return xp.asarray(
        xp.trace(x, offset=offset, dtype=dtype, axis1=-2, axis2=-1, **kwargs)
    )

__all__ = ['cross', 'matmul', 'outer', 'tensordot', 'EighResult',
           'QRResult', 'SlogdetResult', 'SVDResult', 'eigh', 'qr', 'slogdet',
           'svd', 'cholesky', 'matrix_rank', 'pinv', 'matrix_norm',
           'matrix_transpose', 'svdvals', 'vecdot', 'vector_norm', 'diagonal',
           'trace']


def __dir__() -> list[str]:
    return __all__
