from sphinx.ext.autodoc import ModuleLevelDocumenter


class ShortSummaryDocumenter(ModuleLevelDocumenter):
    """An autodocumenter that only renders the short summary of the object."""

    # Defines the usage: .. autoshortsummary:: {{ object }}
    objtype = "shortsummary"

    # Disable content indentation
    content_indent = ""

    # Avoid being selected as the default documenter for some objects, because we are
    # returning `can_document_member` as True for all objects
    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Allow documenting any object."""
        pass

    def get_object_members(self, want_all):
        """Document no members."""
        pass

    def add_directive_header(self, sig):
        """Override default behavior to add no directive header or options."""
        pass

    def add_content(self, more_content):
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        pass


def setup(app):
    app.add_autodocumenter(ShortSummaryDocumenter)
