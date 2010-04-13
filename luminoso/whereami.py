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
        return os.path.dirname(unicode(sys.executable, sys.getfilesystemencoding( )))+os.path.sep+'luminoso'

    return os.path.dirname(unicode(__file__, sys.getfilesystemencoding( )))


# Define the top-level luminoso/ directory, where data files can be found.
package_dir = os.path.dirname(module_path())

def get_icon(path):
    return package_dir + os.path.sep + 'icons' + os.path.sep + path
