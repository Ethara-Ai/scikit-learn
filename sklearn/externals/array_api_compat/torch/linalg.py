from __future__ import annotations

import torch
import torch.linalg

from .._internal import clone_module

__all__ = clone_module("torch.linalg", globals())

# outer is implemented in torch but aren't in the linalg namespace
from torch import outer
from ._aliases import _fix_promotion, sum
# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot
from ._typing import Array, DType
from ..common._typing import JustInt, JustFloat

# Note: torch.linalg.cross does not default to axis=-1 (it defaults to the
# first axis with size 3), see https://github.com/pytorch/pytorch/issues/58743

# torch.cross also does not support broadcasting when it would add new
# dimensions https://github.com/pytorch/pytorch/issues/39656
def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    pass

def vecdot(x1: Array, x2: Array, /, *, axis: int = -1, **kwargs: object) -> Array:
    pass

def solve(x1: Array, x2: Array, /, **kwargs: object) -> Array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    # Torch tries to emulate NumPy 1 solve behavior by using batched 1-D solve
    # whenever
    # 1. x1.ndim - 1 == x2.ndim
    # 2. x1.shape[:-1] == x2.shape
    #
    # See linalg_solve_is_vector_rhs in
    # aten/src/ATen/native/LinearAlgebraUtils.h and
    # TORCH_META_FUNC(_linalg_solve_ex) in
    # aten/src/ATen/native/BatchLinearAlgebra.cpp in the PyTorch source code.
    #
    # The easiest way to work around this is to prepend a size 1 dimension to
    # x2, since x2 is already one dimension less than x1.
    #
    # See https://github.com/pytorch/pytorch/issues/52915
    if x2.ndim != 1 and x1.ndim - 1 == x2.ndim and x1.shape[:-1] == x2.shape:
        x2 = x2[None]
    return torch.linalg.solve(x1, x2, **kwargs)

# torch.trace doesn't support the offset argument and doesn't support stacking
def trace(x: Array, /, *, offset: int = 0, dtype: DType | None = None) -> Array:
    # Use our wrapped sum to make sure it does upcasting correctly
    return sum(torch.diagonal(x, offset=offset, dim1=-2, dim2=-1), axis=-1, dtype=dtype)

def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    # JustFloat stands for inf | -inf, which are not valid for Literal
    ord: JustInt | JustFloat = 2,
    **kwargs: object,
) -> Array:
    # torch.vector_norm incorrectly treats axis=() the same as axis=None
    pass

__all__ += ['outer', 'matmul', 'matrix_transpose', 'tensordot',
            'cross', 'vecdot', 'solve', 'trace', 'vector_norm']

def __dir__() -> list[str]:
    return __all__
