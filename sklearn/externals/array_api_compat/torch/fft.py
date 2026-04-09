from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch  # noqa: F401
import torch.fft

from ._typing import Array
from .._internal import clone_module

__all__ = clone_module("torch.fft", globals())

# Several torch fft functions do not map axes to dim

def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    pass

def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    pass

def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    pass

def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs: object,
) -> Array:
    pass

def fftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] = None,
    **kwargs: object,
) -> Array:
    pass

def ifftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] = None,
    **kwargs: object,
) -> Array:
    pass


__all__ += ["fftn", "ifftn", "rfftn", "irfftn", "fftshift", "ifftshift"]

def __dir__() -> list[str]:
    return __all__
