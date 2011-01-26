#!/usr/bin/env python
VERSION = "1.3.2"

from setuptools import setup, find_packages
import os.path, sys
from stat import ST_MTIME

if 'py2exe' in sys.argv:
    import py2exe

import modulefinder

for p in sys.path:
   modulefinder.AddPackagePath(__name__, p)
sys.path.append('luminoso/lib')

classifiers=[
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: C',
    'Programming Language :: Python :: 2.5',
    'Programming Language :: Python :: 2.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
    'Topic :: Text Processing :: Linguistic',]

INCLUDES = ["sip", "PySide.QtCore", "PySide.Qt", "PySide.QtGui", "PySide",
            'csc.divisi2', 'spyderlib', 'csc.nl',
            'standalone_nlp.lang_en', 'jinja2', 'numpy', 'chardet', 'pysparse']
DATA_FILES = ['icons']

setup(
    name="Luminoso",
    version = VERSION,
    maintainer='MIT Media Lab, Software Agents group',
    maintainer_email='conceptnet@media.mit.edu',     
    url='http://launchpad.net/luminoso/',
    license = "http://www.gnu.org/copyleft/gpl.html",
    platforms = ["any"],
    description = "A Python GUI for semantic analysis using Divisi",
    classifiers = classifiers,
    ext_modules = [],
    packages=find_packages()+['icons', 'study_skel'],
    app=['luminoso/run_luminoso.py'],
    scripts=['luminoso/run_luminoso.py', 'luminoso/study.py'],
    windows=[{'script': 'luminoso/run_luminoso.py'}],
    install_requires=['csc-utils >= 0.5', 'divisi2', 'ipython >= 0.9.1', 'jinja2', 'chardet'],
    package_data={'csc.nl': ['mblem/*.pickle', 'en/*.txt']},
    include_package_data=True,
    #data_files=DATA_FILES,
    options={'py2exe': {
            'skip_archive': True,
            'includes': INCLUDES,
            'excludes': ["Tkconstants","Tkinter","tcl"]
        },
        'py2app': {
            "argv_emulation": True,
            'includes': INCLUDES,
        },
    },

    entry_points={'gui_scripts': ['luminoso = luminoso.run_luminoso:main'],
                  'console_scripts': ['luminoso-study = luminoso.study:main']},
)

'''
Add the icons directory to the build.

Commented out by Rob; this shouldn't be triggered just by importing the setup
script.
'''
#import shutil

##Create paths. One that points to the icons directory and one that creates
##the disired place to copy to.
#
#srcPath = os.path.abspath('./icons')
#targPath = os.path.abspath('./dist/icons')

##Make sure the icons directory exists and that it has not been added to the build
##yet.
#if os.path.exists(srcPath) and (not os.path.exists(targPath)):
#    shutil.copytree(srcPath,targPath)

