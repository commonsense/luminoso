#!/usr/bin/env python
import py2exe

VERSION = "1.0"

from distutils.core import setup
import os.path, sys
from stat import ST_MTIME

import modulefinder
for p in sys.path:
   modulefinder.AddPackagePath(__name__, p)

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
    'Programming Language :: Java',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
    'Topic :: Text Processing :: Linguistic',]

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
    packages=['luminoso'],
    scripts=['luminoso/run_luminoso.py'],
    windows=[{'script': 'luminoso/run_luminoso.py'}],
    install_requires=['csc-utils >= 0.4.2', 'divisi >= 0.6.8', 'conceptnet >= 4.0rc2', 'sip'],
    package_data={'csc.nl': ['mblem/*.pickle', 'en/*.txt'],
       '': ['icons', 'study_skel', 'ThaiFoodStudy']},
    options={'py2exe': {
        'skip_archive': True,
        'includes': ['csc.divisi.tensor', 'csc.nl.euro', 'sip', 'spyderlib', 'simplejson', 'numpy', 'jinja2'],
    }}
)
