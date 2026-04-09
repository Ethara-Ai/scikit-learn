"""Vendored from
https://github.com/pypa/packaging/blob/main/packaging/version.py
"""
# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]

InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[
    NegativeInfinityType,
    Tuple[
        Union[
            SubLocalType,
            Tuple[SubLocalType, str],
            Tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]
CmpKey = Tuple[
    int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
LegacyCmpKey = Tuple[int, Tuple[str, ...]]
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)


def parse(version: str) -> Union["LegacyVersion", "Version"]:
    """Parse the given version from a string to an appropriate class.

    Parameters
    ----------
    version : str
        Version in a string format, eg. "0.9.1" or "1.2.dev0".

    Returns
    -------
    version : :class:`Version` object or a :class:`LegacyVersion` object
        Returned class depends on the given version: if is a valid
        PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:
    _key: Union[CmpKey, LegacyCmpKey]

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        return self._version

    def __repr__(self) -> str:
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        pass

    @property
    def base_version(self) -> str:
        pass

    @property
    def epoch(self) -> int:
        pass

    @property
    def release(self) -> None:
        pass

    @property
    def pre(self) -> None:
        pass

    @property
    def post(self) -> None:
        pass

    @property
    def dev(self) -> None:
        pass

    @property
    def local(self) -> None:
        return None

    @property
    def is_prerelease(self) -> bool:
        pass

    @property
    def is_postrelease(self) -> bool:
        pass

    @property
    def is_devrelease(self) -> bool:
        pass


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)

_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    pass


def _legacy_cmpkey(version: str) -> LegacyCmpKey:

    # We hardcode an epoch of -1 here. A PEP 440 version can only have a epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    pass


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        pass

    @property
    def release(self) -> Tuple[int, ...]:
        pass

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        pass

    @property
    def post(self) -> Optional[int]:
        pass

    @property
    def dev(self) -> Optional[int]:
        pass

    @property
    def local(self) -> Optional[str]:
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        pass

    @property
    def base_version(self) -> str:
        pass

    @property
    def is_prerelease(self) -> bool:
        pass

    @property
    def is_postrelease(self) -> bool:
        pass

    @property
    def is_devrelease(self) -> bool:
        pass

    @property
    def major(self) -> int:
        pass

    @property
    def minor(self) -> int:
        pass

    @property
    def micro(self) -> int:
        pass


def _parse_letter_version(
    letter: str, number: Union[str, bytes, SupportsInt]
) -> Optional[Tuple[str, int]]:

    pass


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> Optional[LocalType]:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    pass


def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[Tuple[SubLocalType]],
) -> CmpKey:

    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    pass
