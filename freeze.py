from bbfreeze import Freezer
import os
import shutil
from luminoso.window import VERSION
f = Freezer("dist/luminoso", includes=("sip", "PyQt4", "jinja2", "luminoso.study"), excludes=("django", "nose", "bzrlib", "south"))
f.addScript("luminoso/run_luminoso.py", gui_only=True)
f()

## ding dong the ImportError is dead
#bad_file = os.path.sep.join(['dist', 'luminoso', 'Luminoso-%s-py2.5.egg' % VERSION, 'luminoso', 'luminoso.pyc'])
#os.unlink(bad_file)

useful_dirs = ['icons', 'study_skel', 'ThaiFoodStudy', 'AnalogySpace']
for dir in useful_dirs:
    shutil.copytree(dir, 'dist'+os.path.sep+'luminoso'+os.path.sep+dir)

