import shutil
import sys

import click
from spin.cmds import util


@click.command()
def clean():
    """🪥 Clean build folder.

    Very rarely needed since meson-python recompiles as needed when sklearn is
    imported.

    One known use case where "spin clean" is useful: avoid compilation errors
    when switching from numpy<2 to numpy>=2 in the same conda environment or
    virtualenv.
    """
    pass
