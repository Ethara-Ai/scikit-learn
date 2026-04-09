"""Extract reference documentation from the NumPy source tree."""

import copy
import inspect
import pydoc
import re
import sys
import textwrap
from collections import namedtuple
from collections.abc import Callable, Mapping
from functools import cached_property
from warnings import warn


def strip_blank_lines(l):
    "Remove leading and trailing blank lines from a list of lines"
    pass


class Reader:
    """A line-based string reader."""

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\\n'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split("\n")  # store string as list of lines

        self.reset()

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        pass

    def read(self):
        if not self.eof():
            out = self[self._l]
            self._l += 1
            return out
        else:
            return ""

    def seek_next_non_empty_line(self):
        pass

    def eof(self):
        return self._l >= len(self._str)

    def read_to_condition(self, condition_func):
        pass

    def read_to_next_empty_line(self):
        pass

    def read_to_next_unindented_line(self):
        pass

    def peek(self, n=0):
        pass

    def is_empty(self):
        pass


class ParseError(Exception):
    def __str__(self):
        message = self.args[0]
        if hasattr(self, "docstring"):
            message = f"{message} in {self.docstring!r}"
        return message


Parameter = namedtuple("Parameter", ["name", "type", "desc"])


class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.

    """

    sections = {
        "Signature": "",
        "Summary": [""],
        "Extended Summary": [],
        "Parameters": [],
        "Attributes": [],
        "Methods": [],
        "Returns": [],
        "Yields": [],
        "Receives": [],
        "Other Parameters": [],
        "Raises": [],
        "Warns": [],
        "Warnings": [],
        "See Also": [],
        "Notes": [],
        "References": "",
        "Examples": "",
        "index": {},
    }

    def __init__(self, docstring, config=None):
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split("\n")

        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)

        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location(f"Unknown section {key}", error=False)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        return iter(self._parsed_data)

    def __len__(self):
        return len(self._parsed_data)

    def _is_at_section(self):
        pass

    def _strip(self, doc):
        pass

    def _read_to_next_section(self):
        pass

    def _read_sections(self):
        pass

    def _parse_param_list(self, content, single_element_is_type=False):
        pass

    # See also supports the following formats.
    #
    # <FUNCNAME>
    # <FUNCNAME> SPACE* COLON SPACE+ <DESC> SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)+ (COMMA | PERIOD)? SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)* SPACE* COLON SPACE+ <DESC> SPACE*

    # <FUNCNAME> is one of
    #   <PLAIN_FUNCNAME>
    #   COLON <ROLE> COLON BACKTICK <PLAIN_FUNCNAME> BACKTICK
    # where
    #   <PLAIN_FUNCNAME> is a legal function name, and
    #   <ROLE> is any nonempty sequence of word characters.
    # Examples: func_f1  :meth:`func_h1` :obj:`~baz.obj_r` :class:`class_j`
    # <DESC> is a string describing the function.

    _role = r":(?P<role>(py:)?\w+):"
    _funcbacktick = r"`(?P<name>(?:~\w+\.)?[a-zA-Z0-9_\.-]+)`"
    _funcplain = r"(?P<name2>[a-zA-Z0-9_\.-]+)"
    _funcname = r"(" + _role + _funcbacktick + r"|" + _funcplain + r")"
    _funcnamenext = _funcname.replace("role", "rolenext")
    _funcnamenext = _funcnamenext.replace("name", "namenext")
    _description = r"(?P<description>\s*:(\s+(?P<desc>\S+.*))?)?\s*$"
    _func_rgx = re.compile(r"^\s*" + _funcname + r"\s*")
    _line_rgx = re.compile(
        r"^\s*"
        + r"(?P<allfuncs>"
        + _funcname  # group for all function names
        + r"(?P<morefuncs>([,]\s+"
        + _funcnamenext
        + r")*)"
        + r")"
        + r"(?P<trailing>[,\.])?"  # end of "allfuncs"
        + _description  # Some function lists have a trailing comma (or period)  '\s*'
    )

    # Empty <DESC> elements are replaced with '..'
    empty_description = ".."

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
        pass

    def _parse_index(self, section, content):
        """
        .. index:: default
           :refguide: something, else, and more

        """
        pass

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        pass

    def _parse(self):
        pass

    @property
    def _obj(self):
        pass

    def _error_location(self, msg, error=True):
        pass

    # string conversion routines

    def _str_header(self, name, symbol="-"):
        pass

    def _str_indent(self, doc, indent=4):
        pass

    def _str_signature(self):
        pass

    def _str_summary(self):
        pass

    def _str_extended_summary(self):
        pass

    def _str_param_list(self, name):
        pass

    def _str_section(self, name):
        pass

    def _str_see_also(self, func_role):
        pass

    def _str_index(self):
        pass

    def __str__(self, func_role=""):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        out += self._str_param_list("Parameters")
        for param_list in ("Attributes", "Methods"):
            out += self._str_param_list(param_list)
        for param_list in (
            "Returns",
            "Yields",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
        ):
            out += self._str_param_list(param_list)
        out += self._str_section("Warnings")
        out += self._str_see_also(func_role)
        for s in ("Notes", "References", "Examples"):
            out += self._str_section(s)
        out += self._str_index()
        return "\n".join(out)


def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    pass


class FunctionDoc(NumpyDocString):
    def __init__(self, func, role="func", doc=None, config=None):
        self._f = func
        self._role = role  # e.g. "func" or "meth"

        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ""
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config)

    def get_func(self):
        pass

    def __str__(self):
        out = ""

        func, func_name = self.get_func()

        roles = {"func": "function", "meth": "method"}

        if self._role:
            if self._role not in roles:
                print(f"Warning: invalid role {self._role}")
            out += f".. {roles.get(self._role, '')}:: {func_name}\n    \n\n"

        out += super().__str__(func_role=self._role)
        return out


class ObjDoc(NumpyDocString):
    def __init__(self, obj, doc=None, config=None):
        self._f = obj
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config=config)


class ClassDoc(NumpyDocString):
    extra_public_methods = ["__call__"]

    def __init__(self, cls, doc=None, modulename="", func_doc=FunctionDoc, config=None):
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError(f"Expected a class or None, but got {cls!r}")
        self._cls = cls

        if "sphinx" in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        if config is None:
            config = {}
        self.show_inherited_members = config.get("show_inherited_class_members", True)

        if modulename and not modulename.endswith("."):
            modulename += "."
        self._mod = modulename

        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        NumpyDocString.__init__(self, doc)

        _members = config.get("members", [])
        if _members is ALL:
            _members = None
        _exclude = config.get("exclude-members", [])

        if config.get("show_class_members", True) and _exclude is not ALL:

            def splitlines_x(s):
                pass

            for field, items in [
                ("Methods", self.methods),
                ("Attributes", self.properties),
            ]:
                if not self[field]:
                    doc_list = []
                    for name in sorted(items):
                        if name in _exclude or (_members and name not in _members):
                            continue
                        try:
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(Parameter(name, "", splitlines_x(doc_item)))
                        except AttributeError:
                            pass  # method doesn't exist
                    self[field] = doc_list

    @property
    def methods(self):
        pass

    @property
    def properties(self):
        pass

    @staticmethod
    def _should_skip_member(name, klass):
        pass

    def _is_show_member(self, name):
        pass


def get_doc_object(
    obj,
    what=None,
    doc=None,
    config=None,
    class_doc=ClassDoc,
    func_doc=FunctionDoc,
    obj_doc=ObjDoc,
):
    pass
