from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

from ._typing import Array, Device, DType, Namespace

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]

# Note: NumPy fft functions improperly upcast float32 and complex64 to
# complex128, which is why we require wrapping them all here.

def fft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    res = xp.fft.fft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    res = xp.fft.ifft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    pass

def ifftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    pass

def rfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    pass

def irfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    pass

def rfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    pass

def irfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    pass

def hfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    pass

def ihfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    pass

def fftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    pass

def rfftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    pass

def fftshift(
    x: Array, /, xp: Namespace, *, axes: int | Sequence[int] | None = None
) -> Array:
    pass

def ifftshift(
    x: Array, /, xp: Namespace, *, axes: int | Sequence[int] | None = None
) -> Array:
    pass

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

def __dir__() -> list[str]:
    return __all__
