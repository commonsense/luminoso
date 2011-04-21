"""
Utility functions to determine where on the filesystem this code is being run
from.
"""

import os
import sys

def we_are_frozen():
    """Returns whether we are frozen via py2exe.
    This will affect how we find out where we are located."""

    return hasattr(sys, "frozen")

def module_path():
    """ This will get us the program's directory,
    even if we are frozen using py2exe"""

    if we_are_frozen():
        return os.path.dirname(
          unicode(sys.executable, sys.getfilesystemencoding())
        ) + os.path.sep + 'luminoso'

    return os.path.dirname(unicode(__file__, sys.getfilesystemencoding()))

# Define the top-level luminoso/ directory, where data files can be found.
PACKAGE_DIR = os.path.dirname(module_path())

def get_project_filename(path):
    """
    Get a valid path referring to a given filename in the project directory.
    """
    if PACKAGE_DIR:
        return PACKAGE_DIR + os.path.sep + path
    else:
        return path

def get_icon(path):
    """
    Get a filename in the top-level `icons` directory.
    """
    return PACKAGE_DIR + os.path.sep + 'icons' + os.path.sep + path
